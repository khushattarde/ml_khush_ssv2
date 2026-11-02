# `embeddings/` — cached frame embeddings

This folder holds the precomputed per-frame embeddings for every encoder.
**All `.npy` files are gitignored** (each encoder is ~2 GB across the three
arrays; 4 encoders × 2 GB ≈ 8 GB total).

## Layout after `extract_embeddings.py` runs

```
embeddings/
├── dino_cls.npy            (N_clips, 16, 768)      ← DINOv2 CLS token per frame
├── dino_patchmean.npy      (N_clips, 16, 768)      ← mean of patch tokens
├── dino_grid2x2.npy        (N_clips, 16, 4, 768)   ← 2×2 spatial pooling
├── clip_cls.npy            (N_clips, 16, 512)
├── clip_patchmean.npy      (N_clips, 16, 512)
├── clip_grid2x2.npy        (N_clips, 16, 4, 512)
├── siglip_cls.npy          (N_clips, 16, 768)
├── siglip_patchmean.npy    (N_clips, 16, 768)
├── siglip_grid2x2.npy      (N_clips, 16, 4, 768)
├── internvit_cls.npy       (N_clips, 16, 3072)
├── internvit_patchmean.npy (N_clips, 16, 3072)
└── internvit_grid2x2.npy   (N_clips, 16, 4, 3072)
```

`N_clips` is the **full** dataset length (including invalid rows) so row
indices stay aligned with `data/frames.npy` and `results/dataset_index.csv`.
Invalid rows contain zeros and are filtered at load time.

## How to populate

```bash
python scripts/extract_embeddings.py                    # all 4 encoders
python scripts/extract_embeddings.py --encoders dino    # just DINOv2
python scripts/extract_embeddings.py --analyze          # + diagnostic figures
```

## What each array means

| file              | per-frame content                                          | used by                       |
|-------------------|------------------------------------------------------------|-------------------------------|
| `<enc>_cls.npy`   | CLS token (DINOv2/CLIP/InternViT) or attention-pooled head (SigLIP) — the encoder's "global" summary of one frame | training, evaluation, cosine analysis |
| `<enc>_patchmean.npy` | mean of all patch tokens — a different "global" summary | embedding analysis (§3.4)     |
| `<enc>_grid2x2.npy`   | 4 vectors = 2×2 spatial pooling of patch tokens — coarse layout | training (concatenated with CLS) |

The trained models in `src/models.py` consume **`cls + grid2x2.flatten()`**
as their per-frame input (`src/dataset.py:load_features`). `patchmean` is kept
for the diagnostic figures but is not fed to the classifiers.

## fp16 + overflow guard

Extraction runs in `torch.float16` by default for speed. If a batch produces
non-finite CLS values (rare, but InternViT at 448px can overflow on a T4),
`src/embeddings.py` switches that encoder to fp32 for the rest of the run.
The cached arrays are always written as `np.float16` regardless — the standard
deviation of these embeddings is ~0.1, so fp16 has more than enough precision.

## Re-extraction

To force re-extraction (e.g. after changing the encoder config), delete the
three `.npy` files for that encoder:

```bash
rm embeddings/dino_*.npy
python scripts/extract_embeddings.py --encoders dino
```
