#!/usr/bin/env python3
"""Stage 5: train every (encoder × model × seed) and save checkpoints.

Usage:
    python scripts/train_models.py [--config scripts/configs/default.yaml]
                                   [--encoders dino]
                                   [--models delta_transformer]
                                   [--seeds 0 1 2]

Checkpoints are written to models/<encoder>_<model>_seed<k>.pt. Existing
checkpoints are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import train_all
from src.utils import get_logger

log = get_logger("train_models")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/default.yaml")
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)

    if args.encoders:
        cfg.train.encoders = tuple(args.encoders)
    if args.models:
        cfg.train.models = tuple(args.models)
    if args.seeds:
        cfg.train.seeds = tuple(args.seeds)

    log.info("training sweep: encoders=%s models=%s seeds=%s",
             cfg.train.encoders, cfg.train.models, cfg.train.seeds)
    paths = train_all(cfg)
    log.info("done — %d checkpoints in %s", len(paths), cfg.resolve(cfg.train.checkpoints_dir))


if __name__ == "__main__":
    main()
