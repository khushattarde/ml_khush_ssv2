#!/usr/bin/env python3
"""Stage 6: evaluate trained models under ordered / shuffled / reversed frame orderings.

Usage:
    python scripts/evaluate_models.py [--config scripts/configs/default.yaml]
                                      [--mode main|robustness]

Modes:
    main        — accuracy + paired bootstrap for every (encoder × model × seed)
                  → results/section4_all_tiers.csv + results/fig_section4_all.png
    robustness  — progressive observation (first/last k frames) + corruption
                  (blur/noise/occlusion/drop) on DINOv2 features
                  → results/section5_*.csv + results/fig_section5_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.evaluate import evaluate_all, paired_bootstrap, predict, metrics_bundle
from src.dataset import load_features, load_labels
from src.utils import get_logger, ensure_dir, mean_std_str, SHORT_CLASSES
from src.train import _instantiate
from src.models import MODELS
import numpy as np
import pandas as pd
import torch

log = get_logger("evaluate_models")


def _run_main(cfg) -> None:
    """Main results table + paired bootstrap."""
    df = evaluate_all(cfg)
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))

    # Bar chart: ordered accuracy for every (encoder, model)
    if not df.empty:
        import matplotlib.pyplot as plt
        main_models = ["proto_single_frame", "proto_mean_pool", "single_frame", "mean_pool",
                       "gru", "tcn", "transformer", "delta_transformer"]
        present = [m for m in main_models if m in df.model.values]
        if present:
            fig, ax = plt.subplots(figsize=(13, 4.2))
            encoders = df.encoder.unique()
            w = 0.8 / max(len(encoders), 1)
            for j, enc in enumerate(encoders):
                v = [float(str(df[(df.encoder == enc) & (df.model == m)].ordered.iloc[0]).split("±")[0])
                     for m in present if not df[(df.encoder == enc) & (df.model == m)].empty]
                ax.bar(np.arange(len(v)) + w * (j - (len(encoders) - 1) / 2), v, w, label=enc)
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=25, ha="right")
            ax.axhline(0.1, color="gray", ls=":")
            ax.set_ylabel("test accuracy")
            ax.legend()
            plt.title("Section 4: single-frame baselines vs temporal models")
            plt.tight_layout()
            plt.savefig(out_dir / "fig_section4_all.png", dpi=150)
            plt.close()
            log.info("wrote %s", out_dir / "fig_section4_all.png")

    # Paired bootstrap for each (encoder, key comparison)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("\n== paired bootstrap (difference in accuracy) ==")
    boot_rows = []
    for enc in cfg.train.encoders:
        feats, _, val_idx = load_features(enc, cfg)
        X_val = torch.as_tensor(feats, device=device)[torch.as_tensor(val_idx, device=device)]
        y, _, _ = load_labels(cfg)
        y_val = y[val_idx]
        # correctness per clip, averaged over seeds, for each model
        corr: dict[str, np.ndarray] = {}
        for model_name in cfg.train.models:
            cls, _ = MODELS[model_name]
            preds = []
            for seed in cfg.train.seeds:
                ckpt = cfg.resolve(cfg.train.checkpoints_dir) / f"{enc}_{model_name}_seed{seed}.pt"
                if not ckpt.exists():
                    continue
                m = _instantiate(cls, X_val.shape[-1], d_model=cfg.train.d_model, dropout=cfg.train.dropout).to(device)
                m.load_state_dict(torch.load(ckpt, map_location=device))
                preds.append(predict(m, X_val, "ordered", seed=seed))
                del m
            if preds:
                corr[model_name] = (np.stack(preds) == y_val[None, :]).astype(float).mean(0)

        comparisons = [
            ("temporal vs single frame", "delta_transformer", "single_frame"),
            ("order vs more frames", "delta_transformer", "mean_pool"),
            ("uses order? delta_transformer vs its shuffle-twin",
             "delta_transformer", "delta_transformer_shuftrain"),
            ("uses order? transformer vs its shuffle-twin",
             "transformer", "transformer_shuftrain"),
            ("uses order? gru vs its shuffle-twin", "gru", "gru_shuftrain"),
            ("uses order? tcn vs its shuffle-twin", "tcn", "tcn_shuftrain"),
        ]
        log.info(f"\n[{enc}]")
        for label, a, b in comparisons:
            if a in corr and b in corr:
                d, lo, hi = paired_bootstrap(corr[a], corr[b], n_samples=cfg.eval.bootstrap_samples)
                log.info("  %s: %+.3f [%+.3f, %+.3f]", label, d, lo, hi)
                boot_rows.append(dict(encoder=enc, comparison=label, diff=d, ci_low=lo, ci_high=hi))
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(out_dir / "section4_paired_bootstrap.csv", index=False)
        log.info("wrote %s", out_dir / "section4_paired_bootstrap.csv")


def _run_robustness(cfg) -> None:
    """Progressive observation + corruption on DINOv2 features."""
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from src.embeddings import _load_model, ENCODERS
    from src.data import load_index

    enc = "dino"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(cfg.resolve(cfg.eval.results_dir))
    T = cfg.data.n_frames

    feats, _, val_idx = load_features(enc, cfg)
    X_all = torch.as_tensor(feats, device=device)
    X_val = X_all[torch.as_tensor(val_idx, device=device)]
    y, _, _ = load_labels(cfg)
    y_val = y[val_idx]
    yte_t = torch.as_tensor(y_val, device=device)
    pair_t = torch.as_tensor(y_val < 8, device=device)

    names5 = ["single_frame", "mean_pool", "gru", "tcn", "transformer", "delta_transformer"]

    @torch.no_grad()
    def eval_models(Xe: torch.Tensor) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for n in names5:
            cls, _ = MODELS[n]
            a, ap = [], []
            for seed in cfg.train.seeds:
                ckpt = cfg.resolve(cfg.train.checkpoints_dir) / f"{enc}_{n}_seed{seed}.pt"
                if not ckpt.exists():
                    continue
                m = _instantiate(cls, X_val.shape[-1], d_model=cfg.train.d_model, dropout=cfg.train.dropout).to(device)
                m.load_state_dict(torch.load(ckpt, map_location=device))
                m.eval()
                preds = []
                for s in range(0, len(Xe), 256):
                    preds.append(m(Xe[s:s + 256]).argmax(1))
                p = torch.cat(preds)
                a.append((p == yte_t).float().mean().item())
                ap.append((p[pair_t] == yte_t[pair_t]).float().mean().item())
                del m
            out[n] = (np.array(a), np.array(ap))
        return out

    # ---------- Progressive observation: first k, last k, evenly-sampled k ----------
    def sample_idx(k: int, T: int = 16) -> np.ndarray:
        return np.unique(np.round((np.arange(k) + 0.5) * T / k - 0.5).astype(int))

    KS = list(cfg.eval.robustness_ks)
    prog = {k: eval_models(X_val[:, torch.as_tensor(sample_idx(k, T), device=device)]) for k in KS}

    plt.figure(figsize=(8, 4.2))
    for n in names5:
        if n not in prog[KS[0]]:
            continue
        mu = np.array([prog[k][n][0].mean() for k in KS])
        sd = np.array([prog[k][n][0].std() for k in KS])
        plt.plot(KS, mu, marker="o", label=n)
        plt.fill_between(KS, mu - sd, mu + sd, alpha=0.15)
    plt.axhline(0.1, color="gray", ls=":")
    plt.xlabel("number of frames k (evenly sampled)")
    plt.ylabel("test accuracy")
    plt.title("Progressive observation")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_section5_progressive.png", dpi=150)
    plt.close()
    log.info("wrote %s", out_dir / "fig_section5_progressive.png")

    tab = pd.DataFrame({n: [f"{prog[k][n][0].mean():.3f}" for k in KS] for n in names5 if n in prog[KS[0]]},
                       index=[f"k={k}" for k in KS])
    tab.to_csv(out_dir / "section5_progressive.csv")
    log.info("wrote %s", out_dir / "section5_progressive.csv")

    # ---------- Corruption ----------
    log.info("running corruption experiments (re-encoding through DINOv2)...")
    FR = np.load(cfg.resolve(cfg.data.frames_path), mmap_mode="r")
    df = load_index(cfg, valid_only=True)
    vidx = df.index.values
    trow = vidx[val_idx]

    spec = ENCODERS["dino"]
    dino = _load_model(spec, device, torch.float16)
    imean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
    istd = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)

    # train stats for re-standardization
    cls_ = np.load(cfg.resolve(cfg.embeddings.cache_dir) / "dino_cls.npy")[vidx].astype(np.float32)
    gr_ = np.load(cfg.resolve(cfg.embeddings.cache_dir) / "dino_grid2x2.npy")[vidx].astype(np.float32).reshape(len(vidx), T, -1)
    flat = np.concatenate([cls_, gr_], -1)[np.where(df.split.values == "train")[0]].reshape(-1, cls_.shape[-1] + gr_.shape[-1])
    mu_t = torch.tensor(flat.mean(0), device=device)
    sd_t = torch.tensor(flat.std(0) + 1e-6, device=device)
    del cls_, gr_, flat

    def c_blur(x): return TF.gaussian_blur(x, kernel_size=25, sigma=6.0)
    def c_noise(x): return (x + 0.2 * torch.randn_like(x)).clamp(0, 1)

    def c_occ(x):
        n = x.shape[0]
        ys = torch.randint(0, 113, (n,), device=device)
        xs = torch.randint(0, 113, (n,), device=device)
        r = torch.arange(224, device=device)
        m = ((r[None, :, None] >= ys[:, None, None]) & (r[None, :, None] < ys[:, None, None] + 112) &
             (r[None, None, :] >= xs[:, None, None]) & (r[None, None, :] < xs[:, None, None] + 112))
        return torch.where(m[:, None], torch.full_like(x, 0.5), x)

    @torch.no_grad()
    def embed_test(corrupt=None, frac=1.0, seed=0):
        torch.manual_seed(seed)
        out = []
        for s in range(0, len(trow), 8):
            ids = trow[s:s + 8]
            b = len(ids)
            x = torch.from_numpy(np.stack([FR[i] for i in ids])).to(device)
            x = x.reshape(b * T, 224, 224, 3).permute(0, 3, 1, 2).float() / 255
            if corrupt is not None:
                mask = (torch.rand(b * T, device=device) < frac)[:, None, None, None]
                x = torch.where(mask, corrupt(x), x)
            h = dino(pixel_values=((x - imean) / istd).half()).last_hidden_state
            g = F.adaptive_avg_pool2d(
                h[:, 1:].float().reshape(b * T, 16, 16, -1).permute(0, 3, 1, 2), 2
            ).flatten(2).permute(0, 2, 1).reshape(b, T, -1)
            out.append(torch.cat([h[:, 0].float().reshape(b, T, -1), g], -1))
        return (torch.cat(out) - mu_t) / sd_t

    def drop_frames(Xe, m, seed=0):
        g = torch.Generator(device=device); g.manual_seed(seed)
        idx = torch.rand(len(Xe), T, device=device, generator=g).topk(m).indices.sort(1).values
        return Xe.gather(1, idx[..., None].expand(-1, -1, Xe.shape[-1]))

    R = {"clean": eval_models(X_val)}
    corruptions = [
        ("blur (all frames)", c_blur, 1.0),
        ("noise (all frames)", c_noise, 1.0),
        ("occlusion (all frames)", c_occ, 1.0),
        ("occlusion (50% of frames)", c_occ, 0.5),
    ]
    for name, fn, fr in corruptions:
        R[name] = eval_models(embed_test(fn, fr))
        log.info("done: %s", name)
    R["drop 50% of frames (8 kept)"] = eval_models(drop_frames(X_val, 8))
    R["drop 75% of frames (4 kept)"] = eval_models(drop_frames(X_val, 4))

    acc = pd.DataFrame({c: {n: f"{R[c][n][0].mean():.3f}±{R[c][n][0].std():.3f}" for n in names5}
                        for c in R if n in R[c]}).T
    acc.to_csv(out_dir / "section5_corruption.csv")
    log.info("wrote %s", out_dir / "section5_corruption.csv")

    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.8 / len(names5)
    for j, n in enumerate(names5):
        if n not in R["clean"]:
            continue
        ax.bar(np.arange(len(R)) + w * (j - (len(names5) - 1) / 2),
               [R[c][n][0].mean() for c in R if n in R[c]],
               w, yerr=[R[c][n][0].std() for c in R if n in R[c]],
               capsize=1.5, label=n)
    ax.set_xticks(range(len(R)))
    ax.set_xticklabels(list(R), rotation=25, ha="right")
    ax.axhline(0.1, color="gray", ls=":")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=7, ncol=3)
    plt.title("Corruption: single-frame vs temporal models")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_section5_corruption.png", dpi=150)
    plt.close()
    log.info("wrote %s", out_dir / "fig_section5_corruption.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/default.yaml")
    ap.add_argument("--mode", choices=["main", "robustness"], default="main")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)

    if args.mode == "main":
        _run_main(cfg)
    else:
        _run_robustness(cfg)


if __name__ == "__main__":
    main()
