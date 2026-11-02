"""Evaluation (README §5).

Three responsibilities:

1. **Main results table** — for every (encoder × model × seed), compute accuracy
   under ``ordered``, ``shuffled``, ``reversed`` frame orderings, plus the
   reversal-flip rate and pair/static decomposition.
2. **Paired bootstrap** — 2 000-resample CI for the difference in accuracy
   between any two conditions.
3. **Tier A zero-shot** — CLIP / SigLIP text-image cosine, averaged over frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import Config
from src.dataset import load_features, load_labels
from src.models import MODELS, build_model, trains_on_shuffled
from src.train import _instantiate
from src.utils import SHORT_CLASSES, PARTNER_INDEX, get_logger, mean_std_str, ensure_dir, set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Predict under a given frame ordering
# ---------------------------------------------------------------------------

def _reorder(x: torch.Tensor, order: str, seed: int = 0) -> torch.Tensor:
    """Reorder frames of a ``(B, T, D)`` tensor.

    * ``ordered``  — no-op
    * ``shuffled`` — per-clip random permutation (seeded)
    * ``reversed`` — reverse the time axis
    """
    if order == "ordered":
        return x
    if order == "shuffled":
        g = torch.Generator(device=x.device).manual_seed(seed)
        perm = torch.rand(x.shape[:2], device=x.device, generator=g).argsort(1)
        return x.gather(1, perm[..., None].expand(-1, -1, x.shape[-1]))
    if order == "reversed":
        return x.flip(1)
    raise ValueError(f"order must be ordered/shuffled/reversed, got {order!r}")


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    X: torch.Tensor,
    order: str = "ordered",
    batch_size: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Return predicted class indices ``(N,)`` for ``X`` under ``order``."""
    model.eval()
    out: list[torch.Tensor] = []
    for s in range(0, len(X), batch_size):
        xb = _reorder(X[s:s + batch_size], order, seed=seed)
        out.append(model(xb).argmax(1))
    return torch.cat(out).cpu().numpy()


# ---------------------------------------------------------------------------
# Metric bundle
# ---------------------------------------------------------------------------

def metrics_bundle(
    pred_ordered: np.ndarray,
    pred_shuffled: np.ndarray,
    pred_reversed: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the standard metric bundle. Each input is ``(seeds, N)`` or ``(N,)``.

    Returns a dict of arrays — use :func:`src.utils.mean_std_str` to format.
    """
    Po = np.atleast_2d(pred_ordered)
    Ps = np.atleast_2d(pred_shuffled)
    Pr = np.atleast_2d(pred_reversed)
    partner = PARTNER_INDEX[y_true]
    pair_mask = y_true < 8  # the 8 reversal-pair classes
    ok = (Po == y_true) & pair_mask[None, :]
    flip = ((Pr == partner[None, :]) & ok).sum(1) / np.maximum(ok.sum(1), 1)
    return {
        "ordered":    (Po == y_true[None, :]).mean(1),
        "shuffled":   (Ps == y_true[None, :]).mean(1),
        "reversed":   (Pr == y_true[None, :]).mean(1),
        "flip":       flip,
        "acc_pair":   (Po[:, pair_mask] == y_true[pair_mask][None, :]).mean(1),
        "acc_static": (Po[:, ~pair_mask] == y_true[~pair_mask][None, :]).mean(1),
    }


# ---------------------------------------------------------------------------
# Evaluate all (encoder × model) from saved checkpoints
# ---------------------------------------------------------------------------

def evaluate_all(cfg: Config) -> pd.DataFrame:
    """Load every checkpoint, predict under all 3 orderings across all seeds,
    write ``results/section4_all_tiers.csv``, return the table as a DataFrame.
    """
    tcfg = cfg.train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))
    rows: list[dict] = []

    for enc in tcfg.encoders:
        feats, _, val_idx = load_features(enc, cfg)
        X_all = torch.as_tensor(feats, device=device)
        X_val = X_all[torch.as_tensor(val_idx, device=device)]
        y, _, _ = load_labels(cfg)
        y_val = y[val_idx]

        for model_name in tcfg.models:
            cls, _ = MODELS[model_name]
            preds_o, preds_s, preds_r = [], [], []
            for seed in tcfg.seeds:
                ckpt = cfg.resolve(tcfg.checkpoints_dir) / f"{enc}_{model_name}_seed{seed}.pt"
                if not ckpt.exists():
                    log.warning("missing checkpoint %s — skipping", ckpt.name)
                    continue
                model = _instantiate(cls, X_val.shape[-1], d_model=tcfg.d_model, dropout=tcfg.dropout).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                preds_o.append(predict(model, X_val, "ordered", seed=seed))
                preds_s.append(predict(model, X_val, "shuffled", seed=seed))
                preds_r.append(predict(model, X_val, "reversed", seed=seed))
                del model
            if not preds_o:
                continue
            m = metrics_bundle(np.stack(preds_o), np.stack(preds_s), np.stack(preds_r), y_val)
            row = dict(
                tier="C trained", encoder=enc, model=model_name,
                params=sum(p.numel() for p in _instantiate(cls, X_val.shape[-1]).parameters()),
                **{k: mean_std_str(v) for k, v in m.items()},
            )
            rows.append(row)
            log.info("C trained | %7s | %28s | ordered %s | shuffled %s | reversed %s",
                     enc, model_name, row["ordered"], row["shuffled"], row["reversed"])

    df = pd.DataFrame(rows)
    out_csv = out_dir / "section4_all_tiers.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv, len(df))
    return df


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over test clips.

    Args:
        correct_a, correct_b: ``(N,)`` boolean arrays (per-clip correctness)
            averaged over seeds before passing in.
    Returns:
        ``(mean_diff, ci_low, ci_high)`` — 95 % CI.
    """
    diff = correct_a.astype(float) - correct_b.astype(float)
    rng = np.random.RandomState(seed)
    N = len(diff)
    boots = np.array([diff[rng.randint(0, N, N)].mean() for _ in range(n_samples)])
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# Tier A: CLIP / SigLIP zero-shot text-image classification
# ---------------------------------------------------------------------------

#: Short class name -> human-readable text prompt (for CLIP/SigLIP text encoder).
ZERO_SHOT_PROMPTS: dict[str, str] = {
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


@torch.no_grad()
def zeroshot_clip(
    cfg: Config,
    encoder: str = "clip",
    agg: str = "mean",
) -> pd.DataFrame:
    """Zero-shot classification using CLIP/SigLIP text-image cosine similarity.

    Loads the **full** CLIP/SigLIP model (vision + text + shared projection)
    and re-encodes frames through ``get_image_features()`` so that image and
    text embeddings live in the same shared space (512-dim for CLIP-base,
    768-dim for SigLIP-base). The cached ``<enc>_cls.npy`` arrays are NOT used
    here because they are the pre-projection vision-transformer output and
    would have a different dimensionality.

    For each clip, cosine similarity between each frame embedding and each of
    the 10 class text embeddings is computed, aggregated over frames
    (``mean`` or ``max``), and the argmax is taken.

    Returns a one-row DataFrame with the same columns as the trained-model rows.
    """
    from transformers import AutoTokenizer
    from src.data import load_index
    from src.embeddings import ENCODERS

    if encoder not in ("clip", "siglip"):
        raise ValueError(f"zeroshot only supports clip/siglip, got {encoder!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = ENCODERS[encoder]
    dtype = torch.float16 if (device == "cuda") else torch.float32

    # Load the FULL model (vision + text + projection) — not just the vision tower
    if encoder == "clip":
        from transformers import CLIPModel
        model_id = "openai/clip-vit-base-patch16"
        m = CLIPModel.from_pretrained(model_id).to(device).eval().to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:  # siglip
        from transformers import SiglipModel
        model_id = "google/siglip-base-patch16-224"
        m = SiglipModel.from_pretrained(model_id).to(device).eval().to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Text embeddings (10 classes) — projected to the shared text-image space
    prompts = [ZERO_SHOT_PROMPTS[c] for c in SHORT_CLASSES]
    tok = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    text_feat = m.get_text_features(**tok)                    # (10, D_shared)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_np = text_feat.float().cpu().numpy()                 # (10, D_shared)

    # Image embeddings — re-encode raw frames through get_image_features()
    # so they land in the SAME shared space as text_feat.
    FR = np.load(cfg.resolve(cfg.data.frames_path), mmap_mode="r")
    df = load_index(cfg, valid_only=True)
    vidx = df.index.values
    T = cfg.data.n_frames
    mean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)

    sims = np.zeros((len(vidx), T, 10), dtype=np.float32)
    cpb = 8  # clips per batch = 128 frames per forward
    for s in tqdm(range(0, len(vidx), cpb), desc=f"zeroshot {encoder}"):
        ids = vidx[s:s + cpb]
        b = len(ids)
        x = torch.from_numpy(np.stack([FR[i] for i in ids])).to(device)
        x = x.reshape(b * T, FR.shape[2], FR.shape[3], 3).permute(0, 3, 1, 2).float() / 255
        x = ((x - mean) / std).to(dtype)
        img_feat = m.get_image_features(pixel_values=x)       # (b*T, D_shared)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # cosine sim with text — (b*T, 10) -> reshape (b, T, 10)
        sim = (img_feat.float() @ text_feat.float().T).cpu().numpy()
        sims[ids] = sim.reshape(b, T, 10)

    del m
    if device == "cuda":
        torch.cuda.empty_cache()

    # aggregate over frames and predict
    pred = sims.mean(1).argmax(1) if agg == "mean" else sims.max(1).argmax(1)

    y = df.cls.map({c: i for i, c in enumerate(SHORT_CLASSES)}).values
    partner = PARTNER_INDEX[y]
    pair_m = y < 8
    ok = (pred == y) & pair_m
    flip = ((pred == partner) & ok).sum() / max(ok.sum(), 1) if ok.sum() else float("nan")

    row = pd.DataFrame([dict(
        tier="A zero-shot", encoder=encoder, model=f"zeroshot_{agg}", params=0,
        ordered=f"{(pred == y).mean():.3f}",
        shuffled=f"{(pred == y).mean():.3f}",   # zero-shot is order-blind by construction
        reversed=f"{(pred == y).mean():.3f}",
        flip=f"{flip:.3f}",
        acc_pair=f"{(pred[pair_m] == y[pair_m]).mean():.3f}",
        acc_static=f"{(pred[~pair_m] == y[~pair_m]).mean():.3f}",
    )])
    return row
