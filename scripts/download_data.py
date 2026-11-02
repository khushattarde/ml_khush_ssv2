#!/usr/bin/env python3
"""Stage 1: download SSv2 video parts + labels zip into data/raw/.

Usage:
    python scripts/download_data.py [--config scripts/configs/default.yaml]

Skips files that already exist and are > 100 KB. Verifies the first video part
is gzip data (not the HTML auth-error page Qualcomm returns without auth).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python scripts/x.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import download_data
from src.utils import get_logger

log = get_logger("download_data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/default.yaml")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    log.info("downloading SSv2 to %s", cfg.resolve(cfg.data.raw_dir))
    download_data(cfg)
    log.info("done")


if __name__ == "__main__":
    main()
