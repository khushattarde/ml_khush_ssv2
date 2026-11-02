"""torch Dataset wrappers over cached frame embeddings.

Embeddings are precomputed (see :mod:`src.embeddings`) and stored as ``.npy``
arrays of shape ``(N_clips, 16, D)``. This module loads them, standardizes with
train-split statistics, and serves (features, label) pairs to the training loop.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config
from src.data import load_index
from src.utils import SHORT_CLASSES, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature loading + standardization
# ---------------------------------------------------------------------------

def load_features(encoder: str, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load + standardize the per-frame feature tensor for ``encoder``.

    Returns ``(feats, train_idx, val_idx)`` where:

    * ``feats`` is a ``(N_valid_clips, 16, D)`` float32 array, standardized with
      **train-split** statistics only (no leakage from val).
    * ``train_idx`` / ``val_idx`` are 1-D int arrays indexing into ``feats``.

    ``D = cls_dim + 4 * cls_dim`` because we concatenate the CLS token with the
    flattened 2×2 spatial grid (same layout as the notebook).
    """
    cache = cfg.resolve(cfg.embeddings.cache_dir)
    df = load_index(cfg, valid_only=True)
    vidx = df.index.values

    cls_path = cache / f"{encoder}_cls.npy"
    grid_path = cache / f"{encoder}_grid2x2.npy"
    if not cls_path.exists() or not grid_path.exists():
        raise FileNotFoundError(
            f"missing {cls_path.name} or {grid_path.name} in {cache} — "
            f"run `python scripts/extract_embeddings.py --encoders {encoder}` first."
        )

    cls = np.load(cls_path)[vidx].astype(np.float32)            # (N, T, D_cls)
    gr = np.load(grid_path)[vidx].astype(np.float32)            # (N, T, 4, D_cls)
    gr = gr.reshape(len(vidx), gr.shape[1], -1)                 # (N, T, 4*D_cls)
    feats = np.concatenate([cls, gr], axis=-1)                  # (N, T, D)

    # split masks
    y = df.cls.map({c: i for i, c in enumerate(SHORT_CLASSES)}).values
    is_train = df.split.values == "train"
    train_idx = np.where(is_train)[0]
    val_idx = np.where(~is_train)[0]

    # standardize with train statistics only
    flat = feats[train_idx].reshape(-1, feats.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0) + 1e-6
    feats = ((feats - mean) / std).astype(np.float32)

    log.info("[%s] features: %s | train=%d val=%d | D=%d",
             encoder, feats.shape, len(train_idx), len(val_idx), feats.shape[-1])
    return feats, train_idx, val_idx


def load_labels(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y, train_idx, val_idx)`` aligned with :func:`load_features`."""
    df = load_index(cfg, valid_only=True)
    y = df.cls.map({c: i for i, c in enumerate(SHORT_CLASSES)}).values
    is_train = df.split.values == "train"
    return y, np.where(is_train)[0], np.where(~is_train)[0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Minimal Dataset over a ``(N, T, D)`` numpy array + integer labels.

    Args:
        feats: ``(N, T, D)`` float32 array.
        labels: ``(N,)`` int64 array of class indices.
        order: one of ``"ordered"``, ``"shuffled"``, ``"reversed"``.
            Shuffled uses a per-clip random permutation (seeded at construction).
        seed: seed for the shuffled permutation.
    """

    def __init__(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        order: str = "ordered",
        seed: int = 0,
    ) -> None:
        if order not in ("ordered", "shuffled", "reversed"):
            raise ValueError(f"order must be ordered/shuffled/reversed, got {order!r}")
        self.feats = feats
        self.labels = labels
        self.order = order
        self._shuf = None
        if order == "shuffled":
            rng = np.random.RandomState(seed)
            self._shuf = np.stack([rng.permutation(feats.shape[1]) for _ in range(len(feats))])

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.feats[idx]
        if self.order == "shuffled":
            x = x[self._shuf[idx]]
        elif self.order == "reversed":
            x = x[::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.as_tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Frame-order manipulation (used by training-time shuffle augmentation)
# ---------------------------------------------------------------------------

def temporal_shuffle(x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Per-clip random temporal shuffle of a ``(B, T, D)`` tensor.

    Different from :class:`EmbeddingDataset`'s ``shuffled`` order in that this
    is meant to be called inside the training loop on each batch (so the
    shuffle changes every epoch).
    """
    if seed is not None:
        g = torch.Generator(device=x.device).manual_seed(seed)
        perm = torch.rand(x.shape[:2], device=x.device, generator=g).argsort(1)
    else:
        perm = torch.rand(x.shape[:2], device=x.device).argsort(1)
    return x.gather(1, perm[..., None].expand(-1, -1, x.shape[-1]))
