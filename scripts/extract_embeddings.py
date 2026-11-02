#!/usr/bin/env python3
"""Stage 4: extract frame embeddings for every configured encoder.

Usage:
    python scripts/extract_embeddings.py [--config scripts/configs/default.yaml]
                                         [--encoders dino clip siglip internvit]
                                         [--analyze]

For each encoder, writes three arrays into embeddings/:
    <enc>_cls.npy        (N_clips, 16, D)      — global per-frame embedding
    <enc>_patchmean.npy  (N_clips, 16, D)      — mean of patch tokens
    <enc>_grid2x2.npy    (N_clips, 16, 4, D)   — 2x2 spatial pooling

With --analyze, also runs the 4 diagnostic figures (cosine by class, sequence
similarity, PCA trajectories, temporal pooling summary).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.embeddings import extract_encoder, extract_all
from src.utils import get_logger

log = get_logger("extract_embeddings")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/default.yaml")
    ap.add_argument("--encoders", nargs="*", default=None,
                    help="subset of encoders to run (default: all in config)")
    ap.add_argument("--analyze", action="store_true",
                    help="also run the 4 diagnostic figures from §3.4")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)

    if args.encoders:
        cfg.embeddings.encoders = tuple(args.encoders)

    log.info("extracting embeddings for: %s", cfg.embeddings.encoders)
    extract_all(cfg)

    if args.analyze:
        log.info("running embedding analysis...")
        from src.visualization import embedding_analysis
        embedding_analysis(cfg)

    log.info("done — see embeddings/*.npy")


if __name__ == "__main__":
    main()
