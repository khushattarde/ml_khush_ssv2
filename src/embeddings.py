"""Frame embedding extraction (README §3).

For each encoder, three arrays are written to ``embeddings/``:

* ``<enc>_cls.npy``       (N_clips, 16, D)  — global per-frame embedding (CLS / attention pool)
* ``<enc>_patchmean.npy`` (N_clips, 16, D)  — mean of patch tokens per frame
* ``<enc>_grid2x2.npy``   (N_clips, 16, 4, D) — 2×2 spatial pooling of patch tokens

``N_clips`` is the full dataset length (including invalid rows) so row indices
stay aligned with ``data/frames.npy`` and ``results/dataset_index.csv``.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import Config
from src.data import load_index
from src.utils import get_logger, ensure_dir

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------

# ImageNet normalization (DINOv2) and CLIP normalization (CLIP, SigLIP-ish, InternViT)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4815, 0.4578, 0.4082)
CLIP_STD = (0.2686, 0.2613, 0.2758)


@dataclass
class EncoderSpec:
    key: str
    model_id: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    loader: str  # "auto" | "clip_vision" | "siglip_vision" | "internvit"
    resolution: int = 224


ENCODERS: dict[str, EncoderSpec] = {
    "dino": EncoderSpec("dino", "facebook/dinov2-base", IMAGENET_MEAN, IMAGENET_STD, "auto"),
    "clip": EncoderSpec("clip", "openai/clip-vit-base-patch16", CLIP_MEAN, CLIP_STD, "clip_vision"),
    "siglip": EncoderSpec("siglip", "google/siglip-base-patch16-224", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "siglip_vision"),
    "internvit": EncoderSpec("internvit", "OpenGVLab/InternViT-300M-448px", CLIP_MEAN, CLIP_STD, "internvit", resolution=448),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(spec: EncoderSpec, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load a frozen vision encoder from HuggingFace."""
    if spec.loader == "auto":
        from transformers import AutoModel
        m = AutoModel.from_pretrained(spec.model_id)
    elif spec.loader == "clip_vision":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained(spec.model_id)
    elif spec.loader == "siglip_vision":
        from transformers import SiglipVisionModel
        m = SiglipVisionModel.from_pretrained(spec.model_id)
    elif spec.loader == "internvit":
        return _load_internvit(spec, device, dtype)
    else:
        raise ValueError(f"unknown loader {spec.loader!r}")
    return m.to(device).eval().to(dtype)


def _load_internvit(spec: EncoderSpec, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """InternViT-300M needs the modeling file shipped with the HF repo + safetensors load."""
    import glob
    import importlib
    import shutil
    import sys

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    src = snapshot_download(spec.model_id)
    pkg = Path("internvit_pkg")
    pkg.mkdir(exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".py"):
            shutil.copy(os.path.join(src, f), pkg)
    (pkg / "__init__.py").touch()
    sys.path.insert(0, str(Path.cwd()))
    importlib.invalidate_caches()
    m_mod = importlib.import_module("internvit_pkg.modeling_intern_vit")
    cfg_ = m_mod.InternVisionConfig.from_pretrained(src)
    cfg_.use_flash_attn = False  # flash-attn is unusable on T4
    model = m_mod.InternVisionModel(cfg_)
    sd: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(f"{src}/*.safetensors")):
        sd.update(load_file(f))
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss:
        raise RuntimeError(f"InternViT load: {len(miss)} missing keys: {miss[:3]}")
    return model.to(device).eval().to(dtype)


# ---------------------------------------------------------------------------
# Forward pass — returns (global_emb [B,D], patch_tokens [B,P,D])
# ---------------------------------------------------------------------------

def _run(model: torch.nn.Module, key: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward pass and return ``(global_emb, patch_tokens)``."""
    if key == "dino":
        h = model(pixel_values=x).last_hidden_state
        return h[:, 0], h[:, 1:]
    if key == "clip":
        o = model(pixel_values=x)
        return o.pooler_output, o.last_hidden_state[:, 1:]
    if key == "siglip":
        o = model(pixel_values=x)
        return o.pooler_output, o.last_hidden_state
    if key == "internvit":
        o = model(x)
        h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        return h[:, 0], h[:, 1:]
    raise ValueError(f"unknown encoder key {key!r}")


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def extract_encoder(key: str, cfg: Config) -> None:
    """Extract + cache all three arrays for one encoder. No-op if all three exist."""
    if key not in ENCODERS:
        raise ValueError(f"unknown encoder {key!r}; choices: {list(ENCODERS)}")
    spec = ENCODERS[key]
    cache = ensure_dir(cfg.resolve(cfg.embeddings.cache_dir))
    paths = {n: cache / f"{key}_{n}.npy" for n in ("cls", "patchmean", "grid2x2")}
    if all(p.exists() for p in paths.values()):
        log.info("[%s] all three arrays cached; skipping", key)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("[%s] CUDA not available — extraction will be very slow on CPU", key)

    df = load_index(cfg, valid_only=False)  # full index, so row indices align with frames.npy
    N = len(df)
    T = cfg.data.n_frames
    FR = np.load(cfg.resolve(cfg.data.frames_path), mmap_mode="r")
    vidx = df.index.values  # all rows; invalid rows get zeros

    cpb = cfg.embeddings.internvit_clips_per_batch if key == "internvit" else cfg.embeddings.clips_per_batch
    dtype = torch.float16 if (cfg.embeddings.use_fp16 and device == "cuda") else torch.float32
    model = _load_model(spec, device, dtype)
    mean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)
    res = spec.resolution

    log.info("[%s] extracting %d clips × %d frames = %d frames | dtype=%s | batch=%d clips",
             key, N, T, N * T, dtype, cpb)

    # output containers — written incrementally
    A: dict[str, np.ndarray] = {}
    D_global: int | None = None

    for s in tqdm(range(0, len(vidx), cpb), desc=key):
        ids = vidx[s:s + cpb]
        b = len(ids)
        x = torch.from_numpy(np.stack([FR[i] for i in ids])).to(device)
        x = x.reshape(b * T, FR.shape[2], FR.shape[3], 3).permute(0, 3, 1, 2).float() / 255
        if res != cfg.data.img_size:
            x = F.interpolate(x, size=res, mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - mean) / std

        with torch.no_grad():
            g, p = _run(model, key, x.to(dtype))
            if cfg.embeddings.overflow_guard and dtype == torch.float16 and not torch.isfinite(g).all():
                log.warning("[%s] fp16 gave non-finite values — switching to fp32 for the rest", key)
                model.float()
                dtype = torch.float32
                g, p = _run(model, key, x.to(dtype))
        g = g.float()
        p = p.float()
        hw = int(round(p.shape[1] ** 0.5))
        gr = F.adaptive_avg_pool2d(
            p.reshape(b * T, hw, hw, -1).permute(0, 3, 1, 2), 2
        )  # (b*T, D, 2, 2)

        if D_global is None:
            D_global = g.shape[-1]
            A = {
                "cls": np.zeros((N, T, D_global), np.float16),
                "patchmean": np.zeros((N, T, D_global), np.float16),
                "grid2x2": np.zeros((N, T, 4, D_global), np.float16),
            }

        A["cls"][ids] = g.reshape(b, T, -1).cpu().numpy()
        A["patchmean"][ids] = p.mean(1).reshape(b, T, -1).cpu().numpy()
        A["grid2x2"][ids] = (
            gr.flatten(2).permute(0, 2, 1).reshape(b, T, 4, -1).cpu().numpy()
        )

    for n, a in A.items():
        np.save(paths[n], a)
        log.info("[%s] wrote %s %s", key, paths[n].name, a.shape)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def extract_all(cfg: Config) -> None:
    """Extract every encoder in ``cfg.embeddings.encoders``. Continues on per-encoder errors."""
    for key in cfg.embeddings.encoders:
        try:
            extract_encoder(key, cfg)
        except Exception as e:
            log.error("[%s] skipped: %s: %s", key, type(e).__name__, str(e)[:300])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
