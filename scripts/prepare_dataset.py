#!/usr/bin/env python3
"""Stage 2: pick the 10-class subset, decode 16 frames per clip, write data/frames.npy + results/dataset_index.csv.

Usage:
    python scripts/prepare_dataset.py [--config scripts/configs/default.yaml]

The memmap is ~4.6 GB for the default 1900-clip subset. Row i of the memmap
aligns with row i of the CSV; never drop rows — always filter by `valid`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import prepare_dataset
from src.utils import get_logger

log = get_logger("prepare_dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/default.yaml")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    df = prepare_dataset(cfg)
    log.info("done — %d clips, %d valid", len(df), int(df.valid.sum()))


if __name__ == "__main__":
    main()
