"""Generative-VLM zero-shot classification (README §6).

Two open-weights VLMs are run as **multiple-choice classifiers** on the same
val clips:

* `Qwen2-VL-2B <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct>`_
* `SmolVLM2-500M <https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct>`_

Each VLM sees 1, 8 (ordered), or 8 (reversed) frames and is asked to pick the
letter (A-J) of the correct action. The answer distribution is recovered by
softmaxing the next-token logits over the 10 letter tokens, averaged over two
random option orderings to wash out position bias.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.config import Config
from src.data import load_index
from src.utils import SHORT_CLASSES, PARTNER_INDEX, get_logger, ensure_dir, set_seed

log = get_logger(__name__)

# Human-readable action phrases used as multiple-choice options.
OPTIONS: dict[str, str] = {
    "opening":   "Opening something",
    "closing":   "Closing something",
    "move_up":   "Moving something up",
    "move_down": "Moving something down",
    "push_L2R":  "Pushing something from left to right",
    "push_R2L":  "Pushing something from right to left",
    "put_into":  "Putting something into something",
    "take_out":  "Taking something out of something",
    "holding":   "Holding something",
    "showing":   "Showing something to the camera",
}
LETTERS = "ABCDEFGHIJ"


# ---------------------------------------------------------------------------
# num2words patch (transformers < 4.46 imports it lazily inside SmolVLM)
# ---------------------------------------------------------------------------

def patch_num2words() -> None:
    """Monkey-patch ``transformers.models.smolvlm.processing_smolvlm.num2words``.

    Without this, loading SmolVLM2 on a minimal environment raises
    ``ImportError: num2words is required``. The function exists in
    ``requirements.txt`` but transformers imports it lazily inside the processor.
    """
    try:
        import transformers.utils.import_utils as iu
        importlib.invalidate_caches()
        if hasattr(iu, "_num2words_available"):
            iu._num2words_available = True
        import transformers.models.smolvlm.processing_smolvlm as sp
        sp.num2words = __import__("num2words").num2words
        log.info("patched num2words into SmolVLM processor")
    except Exception as e:
        log.warning("num2words patch failed (will retry on model load): %s", e)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_vlm(model_id: str, device: str) -> tuple[Any, Any, torch.dtype]:
    """Load a generative VLM and its processor. Returns ``(processor, model, dtype)``."""
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForMultimodalLM as AutoVLM  # type: ignore

    proc = AutoProcessor.from_pretrained(model_id)
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        kw = dict(low_cpu_mem_usage=True, device_map={"": str(device)})
        try:
            model = AutoVLM.from_pretrained(model_id, dtype=dt, **kw)
        except TypeError:
            model = AutoVLM.from_pretrained(model_id, torch_dtype=dt, **kw)
    except (ImportError, ValueError):
        # no accelerate installed — load on CPU then move
        try:
            model = AutoVLM.from_pretrained(model_id, dtype=dt, low_cpu_mem_usage=True)
        except TypeError:
            model = AutoVLM.from_pretrained(model_id, torch_dtype=dt, low_cpu_mem_usage=True)
        model = model.to(device)
    # SmolVLM defaults to image-splitting (4 tiles per image) — turn it off so
    # we get one tile per frame.
    ip = getattr(proc, "image_processor", None)
    if ip is not None and hasattr(ip, "do_image_splitting"):
        ip.do_image_splitting = False
    return proc, model.eval(), dt


# ---------------------------------------------------------------------------
# Prompt building + scoring
# ---------------------------------------------------------------------------

def _build_msgs(n_frames: int, final_question: str) -> list[dict]:
    """Build the chat messages for one VLM call."""
    content: list[dict] = [{
        "type": "text",
        "text": (
            f"These are {n_frames} frame(s) from a short first-person video of a hand "
            "interacting with an object, in chronological order (Frame 1 is earliest). "
            "Left/right are as seen in the images."
        ),
    }]
    for k in range(n_frames):
        content += [{"type": "text", "text": f"Frame {k + 1}:"}, {"type": "image"}]
    content.append({"type": "text", "text": final_question})
    return [{"role": "user", "content": content}]


def _make_inputs(proc, frames: list[np.ndarray], final_question: str, nested: bool, dt: torch.dtype):
    """Tokenize + image-process one VLM call.

    Note: currently unused — ``score`` and ``describe`` build inputs inline so
    they can handle the per-tensor device/dtype casting more carefully. Kept
    as a reference implementation of the prompt-template -> processor pipeline.
    """
    text = proc.apply_chat_template(
        _build_msgs(len(frames), final_question),
        add_generation_prompt=True, tokenize=False,
    )
    pil = [Image.fromarray(np.asarray(f)) for f in frames]
    return proc(text=[text], images=[pil] if nested else pil, return_tensors="pt")


@torch.no_grad()
def score(
    proc, model, frames: list[np.ndarray], order: list[int],
    letter_ids: list[torch.Tensor], nested: bool, dt: torch.dtype, device: str,
) -> np.ndarray:
    """Score one clip under one option ordering. Returns a ``(10,)`` probability vector."""
    opts = "\n".join(f"{LETTERS[j]}. {OPTIONS[SHORT_CLASSES[c]]}" for j, c in enumerate(order))
    question = f"Which action is shown?\n{opts}\nAnswer with the letter only."

    # Build inputs (text + images)
    text = proc.apply_chat_template(_build_msgs(len(frames), question), add_generation_prompt=True, tokenize=False)
    pil = [Image.fromarray(np.asarray(f)) for f in frames]
    inputs = proc(text=[text], images=[pil] if nested else pil, return_tensors="pt")
    # Move to device, casting only float tensors to dt
    inputs = {k: (v.to(device).to(dt) if v.dtype.is_floating_point else v.to(device))
              for k, v in inputs.items()}

    logits = model(**inputs).logits[0, -1].float()  # next-token logits
    # softmax over the 10 letter tokens (each letter has two spellings: "A" and " A")
    letter_logprobs = torch.stack([torch.logsumexp(logits[l], 0) for l in letter_ids])
    p = F.softmax(letter_logprobs, 0).cpu().numpy()

    out = np.zeros(10)
    for j, c in enumerate(order):
        out[c] = p[j]
    return out


@torch.no_grad()
def describe(proc, model, frames: list[np.ndarray], nested: bool, dt: torch.dtype, device: str) -> str:
    """Free-form description of the clip — used for the qualitative table."""
    question = ("In 2 short sentences: what changes between the first and last frame, "
                "and what becomes clear only over time?")
    text = proc.apply_chat_template(_build_msgs(len(frames), question), add_generation_prompt=True, tokenize=False)
    pil = [Image.fromarray(np.asarray(f)) for f in frames]
    inputs = proc(text=[text], images=[pil] if nested else pil, return_tensors="pt")
    inputs = {k: (v.to(device).to(dt) if v.dtype.is_floating_point else v.to(device))
              for k, v in inputs.items()}
    g = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    return proc.batch_decode(g[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_zeroshot(cfg: Config, model_names: list[str] | None = None) -> pd.DataFrame:
    """Run VLM zero-shot for every model in ``cfg.vlm.models`` (or ``model_names``).

    Writes ``results/vlm_results.csv`` and ``results/vlm_vs_embedding.png``.
    Returns the per-(model, condition) accuracy table.
    """
    patch_num2words()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))
    vlm_out = ensure_dir(cfg.resolve(cfg.vlm.output_dir))
    FR = np.load(cfg.resolve(cfg.data.frames_path), mmap_mode="r")
    df = load_index(cfg, valid_only=True)

    y = df.cls.map({c: i for i, c in enumerate(SHORT_CLASSES)}).values
    T = cfg.data.n_frames
    sel8 = np.linspace(0, T - 1, cfg.vlm.n_frames_per_clip).astype(int)
    rng = np.random.RandomState(cfg.vlm.seed)

    # Sample N clips per class
    n_per = cfg.vlm.n_per_class
    sel = np.concatenate([rng.choice(np.where(y == c)[0], n_per, replace=False) for c in range(10)])
    ys = y[sel]
    pairm = ys < 8

    # Two random option orderings to wash out position bias
    o1 = list(rng.permutation(10))
    orders = [o1, o1[::-1]]

    conditions = {
        "single":   lambda fr: [fr[T // 2]],
        "ordered":  lambda fr: [fr[i] for i in sel8],
        "reversed": lambda fr: [fr[i] for i in sel8[::-1]],
    }

    names = list(cfg.vlm.models.keys()) if model_names is None else model_names
    RES: dict[str, dict[str, np.ndarray]] = {}
    QUAL: dict[str, dict[int, dict[str, str]]] = {}

    # Qualitative clips: first clip of each pair class, capped to n_qualitative
    qual_ids = [sel[np.where(ys == c)[0][0]] for c in range(8)][:cfg.vlm.n_qualitative]

    for name in names:
        mid = cfg.vlm.models[name]
        f_scores = out_dir / f"vlm_scores_{name}.npz"
        f_qual = vlm_out / f"qualitative_{name}.json"
        if f_scores.exists() and f_qual.exists():
            z = np.load(f_scores)
            RES[name] = {c: z[c] for c in conditions}
            with open(f_qual) as f:
                QUAL[name] = {int(k): v for k, v in json.load(f).items()}
            log.info("[%s] loaded saved results", name)
            continue

        log.info("[%s] loading model %s", name, mid)
        proc, model, dt = load_vlm(mid, device)
        nested = "smol" in mid.lower()
        tok = proc.tokenizer
        letter_ids = [
            torch.tensor([
                tok.encode(L, add_special_tokens=False)[0],
                tok.encode(" " + L, add_special_tokens=False)[0],
            ], device=device)
            for L in LETTERS
        ]
        RES[name] = {c: np.zeros((len(sel), 10)) for c in conditions}
        for a, i in enumerate(tqdm(sel, desc=name)):
            fr = np.asarray(FR[df.index.values[i]])
            for c, fn in conditions.items():
                frames = fn(fr)
                RES[name][c][a] = np.mean(
                    [score(proc, model, frames, o, letter_ids, nested, dt, device) for o in orders],
                    axis=0,
                )

        QUAL[name] = {
            int(i): {
                "ordered": describe(proc, model, conditions["ordered"](np.asarray(FR[df.index.values[i]])),
                                    nested, dt, device),
                "reversed": describe(proc, model, conditions["reversed"](np.asarray(FR[df.index.values[i]])),
                                     nested, dt, device),
            }
            for i in qual_ids
        }
        np.savez(f_scores, sel=sel, ys=ys, **RES[name])
        with open(f_qual, "w") as f:
            json.dump({str(k): v for k, v in QUAL[name].items()}, f, indent=2)
        del model, proc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------- results table ----------
    rows = []
    for name in RES:
        for c in conditions:
            P = RES[name][c].argmax(1) == ys
            rows.append(dict(
                model=name, cond=c,
                acc=float(P.mean()),
                se=float(np.sqrt(P.mean() * (1 - P.mean()) / len(P))),
                acc_pair=float(P[pairm].mean()),
                acc_static=float(P[~pairm].mean()),
            ))
        Po = RES[name]["ordered"]
        Pr = RES[name]["reversed"]
        ok = (Po.argmax(1) == ys) & pairm
        flip = (Pr.argmax(1)[ok] == PARTNER_INDEX[ys[ok]]).mean() if ok.sum() else float("nan")
        log.info("[%s] reversal flip rate = %.2f (n=%d) | pair clips P(true): ordered %.3f -> reversed %.3f",
                 name, flip, int(ok.sum()),
                 Po[np.arange(len(ys)), ys][pairm].mean(),
                 Pr[np.arange(len(ys)), ys][pairm].mean())

    res = pd.DataFrame(rows).round(3)
    out_csv = out_dir / "vlm_results.csv"
    res.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)
    return res
