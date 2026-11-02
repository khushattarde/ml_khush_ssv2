"""Utility functions: seeding, logging, class/label helpers, small math.

Everything here is framework-agnostic; torch-specific helpers live next to the
modules that use them.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Class taxonomy — single source of truth
# ---------------------------------------------------------------------------

#: The 10 short class names in canonical order.
SHORT_CLASSES: tuple[str, ...] = (
    "opening", "closing", "move_up", "move_down",
    "push_L2R", "push_R2L", "put_into", "take_out",
    "holding", "showing",
)

#: SSv2 template string -> short name.
TARGETS: dict[str, str] = {
    "Opening [something]": "opening",
    "Closing [something]": "closing",
    "Moving [something] up": "move_up",
    "Moving [something] down": "move_down",
    "Pushing [something] from left to right": "push_L2R",
    "Pushing [something] from right to left": "push_R2L",
    "Putting [something] into [something]": "put_into",
    "Taking [something] out of [something]": "take_out",
    "Holding [something]": "holding",
    "Showing [something] to the camera": "showing",
}

#: Time-reversal pairs (the 4 directional pairs; last two classes are static controls).
PAIRS: tuple[tuple[str, str], ...] = (
    ("opening", "closing"),
    ("move_up", "move_down"),
    ("push_L2R", "push_R2L"),
    ("put_into", "take_out"),
)

#: For each class index, the index of its reversal partner (or -1 for static controls).
PARTNER_INDEX: np.ndarray = np.array([1, 0, 3, 2, 5, 4, 7, 6, -1, -1], dtype=np.int64)


def partner_of(cls: int | str | np.ndarray) -> int | np.ndarray:
    """Return the index of the reversal partner of ``cls`` (or -1 for static classes).

    Accepts a single int / short-name string, or an array of ints.
    """
    if isinstance(cls, str):
        cls = SHORT_CLASSES.index(cls)
    return PARTNER_INDEX[cls]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and (if available) PyTorch + CUDA.

    Call this once at the start of every script for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # determinism — at the cost of some throughput
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOGGER_CACHE: dict[str, logging.Logger] = {}


def get_logger(name: str = "ssv2", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that prints to stderr with a clean format.

    Idempotent — calling twice with the same name returns the same logger.
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(h)
    _LOGGER_CACHE[name] = log
    return log


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if it does not exist; return it as ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def repo_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def unit_normalize(a: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize ``a`` along ``axis``."""
    return a / (np.linalg.norm(a, axis=axis, keepdims=True) + eps)


def mean_std_str(a: Sequence[float] | np.ndarray) -> str:
    """Format ``a`` as ``"mean±std"`` with 3 decimals. Singleton arrays print just the mean."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    if a.size == 1:
        return f"{a.item():.3f}"
    return f"{a.mean():.3f}±{a.std():.3f}"
