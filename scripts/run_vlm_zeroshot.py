#!/usr/bin/env python3
"""Stage 7: run generative VLMs as multiple-choice zero-shot classifiers.

Usage:
    python scripts/run_vlm_zeroshot.py [--config scripts/configs/default.yaml]
                                       [--models Qwen2-VL-2B SmolVLM2-500M]

Each VLM sees 1, 8 (ordered), or 8 (reversed) frames and picks the letter of
the correct action. Outputs:
    results/vlm_results.csv          — acc / acc_pair / acc_static / flip per (model, condition)
    vlm/qualitative_<model>.json     — free-form descriptions for 6 example clips
    results/vlm_scores_<model>.npz   — raw probability vectors (for resume)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.vlm import run_zeroshot
from src.utils import get_logger

log = get_logger("run_vlm_zeroshot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/default.yaml")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of VLM names to run (default: all in config)")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)

    log.info("running VLM zero-shot: %s",
             args.models or list(cfg.vlm.models.keys()))
    res = run_zeroshot(cfg, model_names=args.models)
    log.info("done — see results/vlm_results.csv")


if __name__ == "__main__":
    main()
