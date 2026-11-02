"""Training loop (README §4.3).

``fit_one`` trains a single model on a single seed. ``train_all`` sweeps over
(encoder × model × seed) and writes checkpoints to ``models/``.

The recipe is intentionally small (every model is < 1 M parameters and runs on
cached 3840-dim DINOv2 features) so a full sweep of 3 encoders × 10 models × 3
seeds finishes in ~25 min on a T4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.config import Config
from src.dataset import load_features
from src.models import build_model, trains_on_shuffled, count_parameters, MODELS
from src.utils import set_seed, get_logger, ensure_dir, SHORT_CLASSES

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Training a single (encoder, model, seed)
# ---------------------------------------------------------------------------

def _instantiate(model_cls: type[nn.Module], d_in: int, *, d_model: int = 128, dropout: float = 0.3) -> nn.Module:
    """Instantiate ``model_cls`` with the right kwargs for its family."""
    if model_cls.__name__ in ("SingleFrame", "MeanPool"):
        return model_cls(d_in, hidden=256, n_classes=10, dropout=dropout)
    return model_cls(d_in, d_model=d_model, n_classes=10, dropout=0.2)


def fit_one_with_labels(
    model_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    pct_start: float = 0.1,
    d_model: int = 128,
    dropout: float = 0.3,
    seed: int = 0,
    device: str = "cuda",
    verbose: bool = False,
) -> nn.Module:
    """Train one model on ``(X_train, y_train)``.

    If ``model_name`` ends in ``_shuftrain``, every training batch is
    frame-shuffled (per-clip random permutation, fresh every batch).
    """
    set_seed(seed)
    cls, _ = MODELS[model_name]
    train_on_shuffled = trains_on_shuffled(model_name)
    model = _instantiate(cls, X_train.shape[-1], d_model=d_model, dropout=dropout).to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    sched = OneCycleLR(opt, max_lr=lr, total_steps=epochs * n_batches, pct_start=pct_start)

    pbar = tqdm(range(epochs), desc=f"train {model_name} s{seed}", disable=not verbose)
    for _ in pbar:
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        for s in range(0, len(X_train), batch_size):
            idx = perm[s:s + batch_size]
            xb = X_train[idx]
            if train_on_shuffled:
                # per-clip random temporal shuffle, fresh every batch
                p = torch.rand(xb.shape[:2], device=xb.device).argsort(1)
                xb = xb.gather(1, p[..., None].expand(-1, -1, xb.shape[-1]))
            logits = model(xb)
            loss = F.cross_entropy(logits, y_train[idx], label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
        if verbose:
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    return model.eval()


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def train_all(cfg: Config) -> dict[tuple[str, str, int], Path]:
    """Train every (encoder × model × seed) in ``cfg.train`` and save checkpoints.

    Returns a dict ``(encoder, model, seed) -> checkpoint_path``.
    """
    tcfg = cfg.train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = ensure_dir(cfg.resolve(tcfg.checkpoints_dir))
    paths: dict[tuple[str, str, int], Path] = {}

    for enc in tcfg.encoders:
        feats, train_idx, _ = load_features(enc, cfg)
        X = torch.as_tensor(feats, device=device)
        X_train = X[torch.as_tensor(train_idx, device=device)]
        # labels
        from src.dataset import load_labels
        y, _, _ = load_labels(cfg)
        y_train = torch.as_tensor(y[train_idx], device=device, dtype=torch.long)

        for model_name in tcfg.models:
            for seed in tcfg.seeds:
                ckpt = ckpt_dir / f"{enc}_{model_name}_seed{seed}.pt"
                if ckpt.exists():
                    log.info("have %s", ckpt.name)
                    paths[(enc, model_name, seed)] = ckpt
                    continue
                log.info("training %s/%s seed=%d", enc, model_name, seed)
                model = fit_one_with_labels(
                    model_name, X_train, y_train,
                    epochs=tcfg.epochs, batch_size=tcfg.batch_size,
                    lr=tcfg.lr, weight_decay=tcfg.weight_decay,
                    label_smoothing=tcfg.label_smoothing,
                    grad_clip=tcfg.grad_clip, pct_start=tcfg.pct_start,
                    d_model=tcfg.d_model, dropout=tcfg.dropout,
                    seed=seed, device=device, verbose=True,
                )
                torch.save(model.state_dict(), ckpt)
                paths[(enc, model_name, seed)] = ckpt
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
    return paths
