"""Visualization helpers (README §3.4 + §2.3).

Two groups:

* **Dataset-level** — ``dataset_stats``, ``frame_strips_per_class``, ``high_vs_low_change``.
  Run by ``scripts/visualize_dataset.py`` right after ``prepare_dataset``.
* **Embedding-level** — ``embedding_analysis`` (cosine by class, sequence
  similarity matrices, PCA trajectories, temporal pooling summary).
  Run by ``scripts/extract_embeddings.py`` once the cache is populated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config
from src.data import load_index
from src.utils import SHORT_CLASSES, PAIRS, PARTNER_INDEX, get_logger, ensure_dir, unit_normalize

log = get_logger(__name__)

# Resolve Chinese font issues if any (no-op on most systems)
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Dataset-level (Stage 3, run right after prepare_dataset)
# ---------------------------------------------------------------------------

def dataset_stats(cfg: Config) -> None:
    """4-panel summary: clips per class/split, duration, decoded frames, motion by class."""
    df = load_index(cfg, valid_only=True)
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))
    fig, ax = plt.subplots(1, 4, figsize=(20, 3.8))
    df.groupby(["cls", "split"]).size().unstack(fill_value=0).reindex(SHORT_CLASSES).plot.bar(ax=ax[0])
    ax[0].set_title("valid clips per class/split")
    df.dur_s.plot.hist(bins=25, ax=ax[1])
    ax[1].set_title("duration (s)")
    df.n_decoded.plot.hist(bins=25, ax=ax[2])
    ax[2].set_title("decoded frames per clip")
    df.boxplot(column="motion", by="cls", ax=ax[3], rot=90)
    ax[3].set_title("motion score by class")
    plt.suptitle("")
    plt.tight_layout()
    out = out_dir / "dataset_stats.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("wrote %s", out)


def frame_strips_per_class(cfg: Config, n_cols: int = 8) -> None:
    """One row per class, 8 of the 16 frames, from the first train clip of each class."""
    df = load_index(cfg, valid_only=True)
    FR = np.load(cfg.resolve(cfg.data.frames_path), mmap_mode="r")
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))
    T = cfg.data.n_frames
    ts8 = np.linspace(0, T - 1, n_cols).astype(int)
    fig, axes = plt.subplots(len(SHORT_CLASSES), n_cols,
                             figsize=(n_cols * 1.7, len(SHORT_CLASSES) * 1.7))
    for r, c in enumerate(SHORT_CLASSES):
        j = df[(df.cls == c) & (df.split == "train")].index[0]
        for k, t in enumerate(ts8):
            axes[r, k].imshow(FR[j][t])
            axes[r, k].axis("off")
        axes[r, 0].set_title(f"{c}: {df.text[j][:45]}", fontsize=8, loc="left")
    plt.tight_layout()
    out = out_dir / "frame_strips_per_class.png"
    plt.savefig(out, dpi=130)
    plt.close()
    log.info("wrote %s", out)


def high_vs_low_change(cfg: Config, n_cols: int = 8) -> None:
    """Top-4 vs bottom-4 clips by first-last-frame pixel change, 8 frames each."""
    df = load_index(cfg, valid_only=True)
    FR = np.load(cfg.resolve(cfg.data.frames_path), mmap_mode="r")
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))
    T = cfg.data.n_frames
    ts8 = np.linspace(0, T - 1, n_cols).astype(int)
    hi = df.sort_values("first_last", ascending=False).head(4)
    lo = df.sort_values("first_last").head(4)
    fig, axes = plt.subplots(8, n_cols, figsize=(n_cols * 1.7, 8 * 1.7))
    for r, (j, row) in enumerate(list(hi.iterrows()) + list(lo.iterrows())):
        for k, t in enumerate(ts8):
            axes[r, k].imshow(FR[j][t])
            axes[r, k].axis("off")
        axes[r, 0].set_title(
            f"{'HIGH' if r < 4 else 'LOW'} change | {row.cls} | {row.text[:40]}",
            fontsize=8, loc="left",
        )
    plt.tight_layout()
    out = out_dir / "high_vs_low_change.png"
    plt.savefig(out, dpi=130)
    plt.close()
    log.info("wrote %s", out)


# ---------------------------------------------------------------------------
# Embedding-level (Stage 3.4)
# ---------------------------------------------------------------------------

def _background_sim(En: np.ndarray, rng: np.random.RandomState, n: int = 20000) -> float:
    """Mean cosine similarity between random frames of DIFFERENT clips — each encoder's
    'unrelated images' floor, used to rescale raw cosine onto [0, 1]."""
    T = En.shape[1]
    i = rng.randint(0, len(En), n)
    j = rng.randint(0, len(En), n)
    k = i != j
    a = En[i[k], rng.randint(0, T, k.sum())]
    b = En[j[k], rng.randint(0, T, k.sum())]
    return float((a * b).sum(-1).mean())


def _rescale(s: float, bg: float) -> float:
    """0 = as similar as random frames, 1 = identical."""
    return (s - bg) / (1 - bg)


def embedding_analysis(cfg: Config) -> None:
    """Reproduce the 4 diagnostic figures from notebook §3.

    Writes:
      results/fig_cosine_by_class.png
      results/fig_sequence_similarity.png
      results/fig_trajectories_pca.png
      results/fig_temporal_pooling.png
      results/cosine_consecutive_rescaled.csv
      results/sequence_similarity_summary.csv
      results/pooling_summary.csv
    """
    from sklearn.decomposition import PCA

    cache = cfg.resolve(cfg.embeddings.cache_dir)
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))
    df = load_index(cfg, valid_only=True)
    vidx = df.index.values
    lab = df.cls.map({c: i for i, c in enumerate(SHORT_CLASSES)}).values
    T = cfg.data.n_frames
    rng = np.random.RandomState(cfg.data.seed)

    encoders = [k for k in cfg.embeddings.encoders
                if (cache / f"{k}_cls.npy").exists()]
    if not encoders:
        log.warning("no cached embeddings found — skipping embedding analysis")
        return
    log.info("embedding analysis for: %s", encoders)

    EMB = {k: np.load(cache / f"{k}_cls.npy")[vidx].astype(np.float32) for k in encoders}
    EN = {k: unit_normalize(E) for k, E in EMB.items()}

    # ---------- Fig 1: consecutive-frame + first-vs-last cosine by class ----------
    base = {k: _background_sim(En, rng) for k, En in EN.items()}
    cons_n, fl_n = {}, {}
    for k, En in EN.items():
        cons = (En[:, 1:] * En[:, :-1]).sum(-1)            # (N, T-1)
        fl = (En[:, 0] * En[:, -1]).sum(-1)                # (N,)
        cons_n[k] = [_rescale(cons[lab == c].mean(), base[k]) for c in SHORT_CLASSES]
        fl_n[k] = [_rescale(fl[lab == c].mean(), base[k]) for c in SHORT_CLASSES]

    pd.DataFrame(cons_n, index=SHORT_CLASSES).round(4).to_csv(out_dir / "cosine_consecutive_rescaled.csv")
    pd.DataFrame(fl_n, index=SHORT_CLASSES).round(4).to_csv(out_dir / "cosine_first_last_rescaled.csv")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    w = 0.8 / len(encoders)
    for a, (title, d) in zip(ax, [
        ("consecutive-frame similarity (rescaled)", cons_n),
        ("first vs last frame similarity (rescaled)", fl_n),
    ]):
        for j, k in enumerate(encoders):
            a.bar(np.arange(10) + w * (j - (len(encoders) - 1) / 2), d[k], w, label=k)
        a.set_xticks(range(10))
        a.set_xticklabels(SHORT_CLASSES, rotation=60)
        a.set_title(title)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_cosine_by_class.png", dpi=150)
    plt.close()

    # ---------- Fig 2: class-to-class similarity matrices ----------
    ii = {c: i for i, c in enumerate(SHORT_CLASSES)}
    rows = []
    n_cols = len(encoders)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 10.5), squeeze=False)
    for col, k in enumerate(encoders):
        En = EN[k]
        pooled = En.mean(1)
        pooled_c = unit_normalize(pooled - pooled.mean(0))
        delta = En[:, -1] - En[:, 0]
        delta_c = unit_normalize(delta - delta.mean(0))
        cent = unit_normalize(np.stack([pooled_c[lab == c].mean(0) for c in SHORT_CLASSES]))
        dirs = unit_normalize(np.stack([delta_c[lab == c].mean(0) for c in SHORT_CLASSES]))
        Mc, Md = cent @ cent.T, dirs @ dirs.T
        for r, (M, title) in enumerate([
            (Mc, "class similarity (average frame)"),
            (Md, "class similarity (direction of change)"),
        ]):
            a = axes[r, col]
            im = a.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
            a.set_xticks(range(10))
            a.set_yticks(range(10))
            a.set_xticklabels(SHORT_CLASSES, rotation=70, fontsize=7)
            a.set_yticklabels(SHORT_CLASSES, fontsize=7)
            a.set_title(f"{k}: {title}", fontsize=9)
            plt.colorbar(im, ax=a, fraction=0.046)
        nonp = [(i, j) for i in range(8) for j in range(8)
                if i != j and SHORT_CLASSES[j] != [SHORT_CLASSES[p] for p in PARTNER_INDEX[:8]][i]]
        rows.append(dict(
            encoder=k,
            avg_frame_sim_reversal_partners=float(np.mean([Mc[ii[a_], ii[b_]] for a_, b_ in PAIRS])),
            avg_frame_sim_other_classes=float(np.mean([Mc[i, j] for i, j in nonp])),
            change_dir_sim_reversal_partners=float(np.mean([Md[ii[a_], ii[b_]] for a_, b_ in PAIRS])),
            change_dir_sim_other_classes=float(np.mean([Md[i, j] for i, j in nonp])),
        ))
    plt.tight_layout()
    plt.savefig(out_dir / "fig_sequence_similarity.png", dpi=140)
    plt.close()
    pd.DataFrame(rows).round(3).to_csv(out_dir / "sequence_similarity_summary.csv", index=False)

    # ---------- Fig 3: PCA trajectories for reversal pairs ----------
    fig, axes = plt.subplots(len(encoders), 4, figsize=(20, 4.4 * len(encoders)), squeeze=False)
    for r, k in enumerate(encoders):
        E = EMB[k]
        Ec = E - E.mean(1, keepdims=True)
        flat = Ec.reshape(-1, E.shape[-1])
        pca = PCA(2, random_state=cfg.data.seed).fit(flat[rng.choice(len(flat), 20000, replace=False)])
        Z = pca.transform(flat).reshape(len(E), T, 2)
        log.info("[%s] PCA variance explained: %s", k, pca.explained_variance_ratio_.round(3))
        for a, (c1, c2) in zip(axes[r], PAIRS):
            for c, col in [(c1, "tab:blue"), (c2, "tab:red")]:
                zs = Z[lab == c]
                for z in zs[rng.choice(len(zs), 15, replace=False)]:
                    a.plot(z[:, 0], z[:, 1], color=col, alpha=0.15)
                m = zs.mean(0)
                a.plot(m[:, 0], m[:, 1], color=col, lw=3, label=c)
                a.scatter(*m[0], color=col, marker="o", s=70, edgecolor="k")
                a.scatter(*m[-1], color=col, marker=">", s=90, edgecolor="k")
            a.set_title(f"{k}: {c1} vs {c2}  (o=start, >=end)", fontsize=9)
            a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_trajectories_pca.png", dpi=130)
    plt.close()

    # ---------- Fig 4: pooling summary ----------
    prow, orow = [], []
    for k in encoders:
        E = EMB[k]
        Ec = unit_normalize(E - E.reshape(-1, E.shape[-1]).mean(0))
        variants = {
            "single frame (middle)": Ec[:, T // 2],
            "mean pool": unit_normalize(Ec.mean(1)),
            "max pool": unit_normalize(Ec.max(1)),
        }
        same = lab[:, None] == lab[None, :]
        off = ~np.eye(len(lab), dtype=bool)
        for name, X_ in variants.items():
            S = X_ @ X_.T
            prow.append(dict(
                encoder=k, summary=name,
                within_class=float(S[same & off].mean()),
                between_class=float(S[~same].mean()),
                gap=float(S[same & off].mean() - S[~same].mean()),
            ))
        pf = unit_normalize(Ec.mean(1))
        pr = unit_normalize(Ec[:, ::-1].mean(1))
        orow.append(dict(
            encoder=k,
            pooled_vs_pooled_reversed=float((pf * pr).sum(1).mean()),
            first_vs_last_frame=float((Ec[:, 0] * Ec[:, -1]).sum(1).mean()),
        ))
    pool = pd.DataFrame(prow).round(3)
    order = pd.DataFrame(orow).round(3)
    pool.to_csv(out_dir / "pooling_summary.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    pool.pivot(index="summary", columns="encoder", values="gap").loc[
        ["single frame (middle)", "mean pool", "max pool"]
    ].plot.bar(ax=ax[0], rot=15)
    ax[0].set_title("class identity: within-class minus between-class similarity (higher=better)")
    order.set_index("encoder")[["first_vs_last_frame", "pooled_vs_pooled_reversed"]].plot.bar(ax=ax[1], rot=0)
    ax[1].set_title("order test: pooling a clip and its reverse gives the SAME vector (1.0)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_temporal_pooling.png", dpi=150)
    plt.close()
    log.info("embedding analysis done → %s/*.png + *.csv", out_dir)
