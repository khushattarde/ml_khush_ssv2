#!/usr/bin/env python3
"""Stage 3 (optional): dataset statistics + frame strips + high/low-change visualization.

Usage:
    python scripts/visualize_dataset.py [--config scripts/configs/default.yaml]

Outputs (into results/):
    dataset_stats.png            — 4-panel summary (clips per class, duration, frames, motion)
    frame_strips_per_class.png   — 1 row per class, 8 of 16 frames
    high_vs_low_change.png       — top-4 vs bottom-4 clips by first-last pixel change
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.visualization import dataset_stats, frame_strips_per_class, high_vs_low_change
from src.utils import get_logger

log = get_logger("visualize_dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/default.yaml")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    log.info("generating dataset visualizations...")
    dataset_stats(cfg)
    frame_strips_per_class(cfg)
    high_vs_low_change(cfg)
    log.info("done — see results/*.png")


if __name__ == "__main__":
    main()
