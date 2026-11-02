# `models/` — trained classifier checkpoints

This folder holds the trained classifier checkpoints. **All `.pt` files are
gitignored** — each is tiny (< 5 MB; every model has < 1 M parameters because
it runs on top of frozen 3840-dim frame embeddings), but with 3 encoders × 10
models × 3 seeds = 90 files the directory gets noisy.

## Layout after `train_models.py` runs

```
models/
├── dino_single_frame_seed0.pt
├── dino_single_frame_seed1.pt
├── dino_single_frame_seed2.pt
├── dino_mean_pool_seed0.pt
├── ...
├── dino_delta_transformer_shuftrain_seed2.pt
├── clip_single_frame_seed0.pt
├── ...
└── siglip_delta_transformer_shuftrain_seed2.pt
```

Naming convention: `<encoder>_<model>_seed<k>.pt`.

## How to populate

```bash
python scripts/train_models.py                          # full sweep (3 × 10 × 3 = 90)
python scripts/train_models.py --encoders dino          # just DINOv2 (10 × 3 = 30)
python scripts/train_models.py --encoders dino --models delta_transformer --seeds 0 1 2
```

Existing checkpoints are skipped — re-run the script after changing the config
and it will only train the new combinations.

## Loading a checkpoint

```python
import torch
from src.train import _instantiate
from src.models import MODELS

model_name = "delta_transformer"
cls, _ = MODELS[model_name]
model = _instantiate(cls, d_in=3840, d_model=128, dropout=0.3)  # d_in = 768 + 4*768 for DINOv2
model.load_state_dict(torch.load("models/dino_delta_transformer_seed0.pt"))
model.eval()
```

`d_in` depends on the encoder:

| encoder   | cls dim | grid dim (4 × cls) | d_in   |
|-----------|---------|--------------------|--------|
| dino      | 768     | 3072               | 3840   |
| clip      | 512     | 2048               | 2560   |
| siglip    | 768     | 3072               | 3840   |
| internvit | 3072    | 12288              | 15360  |

The dimension is read automatically from the cached embeddings by
`src/dataset.py:load_features`, so you usually don't need to hardcode it.

## Why so small?

Every model in `src/models.py` is a lightweight head on top of frozen frame
embeddings — the heavy lifting (vision encoding) was done once in
`extract_embeddings.py`. This makes the full sweep of 90 runs finish in ~25 min
on a T4, and lets you swap architectures without re-running the encoder.
