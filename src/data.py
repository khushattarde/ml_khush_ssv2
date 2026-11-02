"""SSv2 dataset preparation (README §2).

Pipeline:
    download_data()  → data/raw/20bn-something-something-v2-{00,01} + labels zip
    prepare_dataset() → data/frames.npy (memmap) + results/dataset_index.csv
    visualize()      → results/dataset_stats.png + frame_strips_per_class.png

All functions take a :class:`src.config.Config` so paths/hyper-params are overridable.
"""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm

from src.config import Config
from src.utils import TARGETS, SHORT_CLASSES, get_logger, set_seed, ensure_dir

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_data(cfg: Config) -> None:
    """Download SSv2 video parts + labels zip into ``cfg.data.raw_dir``.

    Skips files that already exist and are > 100 KB (the labels zip is ~5 MB;
    each video part is ~9 GB). Verifies the first part is gzip data, not HTML.
    """
    dcfg = cfg.data
    dest = ensure_dir(cfg.resolve(dcfg.raw_dir))

    jobs = [(dcfg.labels_url, "20bn-something-something-download-package-labels.zip")]
    for p in dcfg.parts:
        jobs.append((dcfg.video_url_tmpl.format(part=p), f"20bn-something-something-v2-{p}"))

    for url, name in jobs:
        out = dest / name
        if out.exists() and out.stat().st_size > 1e5:
            log.info("have %s", name)
            continue
        log.info("downloading %s", name)
        r = subprocess.run(
            f"wget -c -q --show-progress -O '{out}' '{url}'",
            shell=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"wget failed for {name} (code {r.returncode})")

    # sanity check: first video part should be gzip, not HTML (Qualcomm returns HTML on auth errors)
    first_part = dest / f"20bn-something-something-v2-{dcfg.parts[0]}"
    if first_part.exists():
        r = subprocess.run(f"head -c 300 '{first_part}' | file -", shell=True, capture_output=True, text=True)
        log.info("first part file type: %s", r.stdout.strip())
        if "gzip" not in r.stdout:
            raise RuntimeError(
                f"{first_part.name} is not gzip data — the Qualcomm endpoint may require authentication. "
                f"Got: {r.stdout.strip()}"
            )


# ---------------------------------------------------------------------------
# 2. Subset selection + frame decoding
# ---------------------------------------------------------------------------

def _extract_labels(cfg: Config) -> tuple[list[dict], list[dict]]:
    """Unzip the labels package if needed and return (train_json, val_json) as lists."""
    dcfg = cfg.data
    raw = cfg.resolve(dcfg.raw_dir)
    pkg = cfg.resolve(dcfg.labels_pkg_dir)

    lab_zip = [
        p for p in glob.glob(str(raw / "*"))
        if "label" in os.path.basename(p).lower() and zipfile.is_zipfile(p)
    ]
    if not lab_zip:
        raise FileNotFoundError(f"no labels zip found in {raw}")
    if not glob.glob(str(pkg / "**" / "train.json"), recursive=True):
        log.info("extracting labels to %s", pkg)
        ensure_dir(pkg)
        with zipfile.ZipFile(lab_zip[0]) as z:
            z.extractall(pkg)

    def _find(name: str) -> str:
        hits = glob.glob(str(pkg / "**" / name), recursive=True)
        if not hits:
            raise FileNotFoundError(f"{name} not found under {pkg}")
        return hits[0]

    with open(_find("train.json")) as f:
        train = json.load(f)
    with open(_find("validation.json")) as f:
        val = json.load(f)
    return train, val


def pick_subset(train_json: list[dict], val_json: list[dict], cfg: Config) -> pd.DataFrame:
    """Pick ``max_train`` / ``max_val`` clips per target class.

    Returns a DataFrame with columns ``id, cls, split, text``. ``id`` is the SSv2
    numeric id (str); the same id is used as the .webm filename inside the tar.
    """
    dcfg = cfg.data

    def _pick(items: list[dict], cap: int, split: str) -> list[dict]:
        rows: list[dict] = []
        for tpl, short in TARGETS.items():
            ids = [it for it in items if it["template"] == tpl]
            random.Random(dcfg.seed).shuffle(ids)
            rows += [dict(id=it["id"], cls=short, split=split, text=it["label"]) for it in ids[:cap]]
        return rows

    df = pd.DataFrame(
        _pick(train_json, dcfg.max_train_per_class, "train")
        + _pick(val_json, dcfg.max_val_per_class, "val")
    )
    log.info("selected %d clips (%d train, %d val)",
             len(df), (df.split == "train").sum(), (df.split == "val").sum())
    return df


def extract_videos(df: pd.DataFrame, cfg: Config) -> None:
    """Stream-extract ONLY the selected videos from the concatenated tar.gz.

    Reads the ~19 GB tar stream once and writes ``<raw>/20bn-something-something-v2/<id>.webm``
    for every id in ``df``. Skips the extraction if every video already exists.
    """
    dcfg = cfg.data
    raw = cfg.resolve(dcfg.raw_dir)
    vroot = raw / "20bn-something-something-v2"

    have = {p.stem for p in glob.glob(str(vroot / "*.webm"))} if vroot.exists() else set()
    if set(df.id.astype(str)) <= have:
        log.info("all %d videos already extracted", len(df))
        return

    parts = sorted(glob.glob(str(raw / "20bn-something-something-v2-??")))
    if not parts:
        raise FileNotFoundError(f"no SSv2 tar parts in {raw}")
    members_txt = raw / "members.txt"
    members_txt.write_text("\n".join(f"20bn-something-something-v2/{i}.webm" for i in df.id))
    log.info("extracting (reads ~19 GB stream once)...")
    subprocess.run(
        f"cat {' '.join(parts)} | tar -xzf - -C {raw} -T {members_txt}",
        shell=True,
        check=True,
    )


def load_clip(path: str, n: int = 16, size: int = 224) -> tuple[np.ndarray | None, np.ndarray | None, dict, list]:
    """Decode a webm clip with OpenCV (sequential — webm seeking is unreliable),
    uniformly sample ``n`` frames, squash-resize each to ``size×size`` RGB.

    Returns ``(frames, sampled_indices, meta, timestamps)``. If the clip has
    fewer than ``n`` decoded frames, returns ``(None, None, meta, ts)``.
    """
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    ts: list[float] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    meta = dict(
        n_decoded=len(frames),
        fps=fps,
        h=frames[0].shape[0] if frames else 0,
        w=frames[0].shape[1] if frames else 0,
    )
    if len(frames) < n:
        return None, None, meta, ts
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    sampled = np.stack([
        cv2.cvtColor(cv2.resize(frames[i], (size, size)), cv2.COLOR_BGR2RGB)
        for i in idx
    ])
    return sampled, idx, meta, ts


def integrity_ok(idx: np.ndarray, ts: list[float], n: int = 16) -> bool:
    """A clip is valid iff it has exactly ``n`` strictly-increasing sampled indices
    AND monotonically-nondecreasing timestamps. ~10 / 1900 clips fail this."""
    if len(idx) != n:
        return False
    if not bool(np.all(np.diff(idx) > 0)):
        return False
    return all(b >= a for a, b in zip(ts, ts[1:]))


def motion_score(frames: np.ndarray) -> float:
    """Mean absolute pixel difference between consecutive sampled frames (grayscale)."""
    g = frames.astype(np.float32).mean(-1)
    return float(np.mean([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)]))


def prepare_dataset(cfg: Config) -> pd.DataFrame:
    """End-to-end Stage 2: pick subset → extract videos → decode frames → memmap → CSV.

    Returns the dataset_index DataFrame (also written to ``cfg.data.index_csv``).
    """
    dcfg = cfg.data
    set_seed(dcfg.seed)

    train_json, val_json = _extract_labels(cfg)
    df = pick_subset(train_json, val_json, cfg)
    extract_videos(df, cfg)

    vroot = cfg.resolve(dcfg.raw_dir) / "20bn-something-something-v2"
    df["path"] = [str(vroot / f"{i}.webm") for i in df.id]
    missing = [p for p in df.path if not os.path.exists(p)]
    if missing:
        log.warning("extraction missed %d videos; dropping them", len(missing))
    df = df[df.path.map(os.path.exists)].reset_index(drop=True)

    frames_path = cfg.resolve(dcfg.frames_path)
    ensure_dir(frames_path.parent)
    X = open_memmap(
        str(frames_path), mode="w+",
        dtype=np.uint8,
        shape=(len(df), dcfg.n_frames, dcfg.img_size, dcfg.img_size, 3),
    )
    recs: list[dict] = []
    for i, r in tqdm(df.iterrows(), total=len(df), desc="decoding"):
        fr, idx, meta, ts = load_clip(r.path, dcfg.n_frames, dcfg.img_size)
        ok = fr is not None and integrity_ok(idx, ts, dcfg.n_frames)
        if ok:
            X[i] = fr
        recs.append(dict(
            **meta, valid=ok,
            motion=motion_score(fr) if ok else np.nan,
            first_last=float(np.abs(fr[0].astype(np.float32) - fr[-1].astype(np.float32)).mean()) if ok else np.nan,
        ))
    X.flush()
    del X

    df = pd.concat([df, pd.DataFrame(recs)], axis=1)
    df["dur_s"] = df.n_decoded / df.fps
    out_csv = cfg.resolve(dcfg.index_csv)
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    log.info("wrote %s | clips: %d | valid: %d | dropped: %d",
             out_csv, len(df), int(df.valid.sum()), int((~df.valid).sum()))
    return df


# ---------------------------------------------------------------------------
# 3. Loaders used by downstream stages
# ---------------------------------------------------------------------------

def load_index(cfg: Config, valid_only: bool = True) -> pd.DataFrame:
    """Load ``dataset_index.csv`` and optionally filter to valid clips.

    The DataFrame's index is preserved (it addresses the frames memmap row).
    """
    df = pd.read_csv(cfg.resolve(cfg.data.index_csv))
    if valid_only:
        df = df[df.valid]
    return df
