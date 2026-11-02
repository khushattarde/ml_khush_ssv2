# SSv2 Action Recognition — Temporal Order Matters

A reproducible pipeline for **action recognition on a 10-class subset of Something-Something-V2 (SSv2)**,
built around the question:

> *Do frozen image-pretrained encoders (DINOv2 / CLIP / SigLIP / InternViT) already carry enough temporal
> structure to discriminate time-reversal pairs (opening vs closing, push L→R vs R→L, put-into vs take-out),
> or does a lightweight temporal head on top of frame embeddings actually use frame **order**?*

The project ships six end-to-end stages — data prep, embedding extraction, embedding analysis,
single-frame vs temporal model training, robustness / progressive-observation evaluation, and
generative-VLM zero-shot classification — all driven by YAML configs and runnable as plain Python scripts.

```
ssv2-action-recognition/
├── README.md                  ← you are here
├── requirements.txt
├── src/                       ← importable library code
│   ├── config.py              ← dataclass config loaded from YAML
│   ├── data.py                ← download SSv2, pick subset, decode frames to memmap
│   ├── dataset.py             ← torch Dataset over cached embeddings
│   ├── embeddings.py          ← DINOv2 / CLIP / SigLIP / InternViT extraction
│   ├── models.py              ← SingleFrame, MeanPool, GRU, TCN, Transformer, DeltaTransformer
│   ├── train.py               ← AdamW + OneCycleLR training loop
│   ├── evaluate.py            ← ordered / shuffled / reversed eval, paired bootstrap
│   ├── vlm.py                 ← Qwen2-VL / SmolVLM2 multiple-choice zero-shot
│   ├── visualization.py       ← dataset stats, frame strips, PCA trajectories
│   └── utils.py               ← seeding, logging, metrics, IoU helpers
├── scripts/                   ← CLI entry points (one per stage)
│   ├── configs/
│   │   ├── default.yaml       ← single source of truth for all hyper-params
│   │   ├── data.yaml
│   │   └── train.yaml
│   ├── download_data.py
│   ├── prepare_dataset.py
│   ├── extract_embeddings.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── run_vlm_zeroshot.py
│   └── visualize_dataset.py
├── data/                      ← SSv2 raw videos + labels + extracted frames (gitignored)
├── embeddings/                ← cached .npy embedding arrays (gitignored, ~2 GB)
├── models/                    ← trained checkpoints (gitignored)
├── evaluation/                ← evaluation reports + confusion matrices
├── vlm/                       ← VLM prompts, scores, qualitative outputs
├── results/                   ← CSVs + PNGs produced by every stage (committed)
└── notebooks/                 ← original exploratory Colab notebook (reference only)
```

---

## 1. Environment setup

**Hardware.** A single NVIDIA GPU with ≥ 16 GB VRAM is required for embedding extraction and VLM inference
(tested on a Colab T4). Training itself is light — every model in `src/models.py` has < 1 M parameters
because it runs on top of frozen frame embeddings, so a laptop GPU is enough for `scripts/train_models.py`.

**Software.** Python 3.10+.

```bash
# 1. clone
git clone <your-fork-url>/ssv2-action-recognition.git
cd ssv2-action-recognition

# 2. virtualenv
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 3. install (PyTorch matched to your CUDA version — example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. (optional) log in to HuggingFace for InternViT weights
huggingface-cli login
```

Verify the install:

```bash
python -c "import torch, transformers, cv2; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
```

All paths in the code are **relative to the repo root** by default (`data/`, `embeddings/`, `models/`,
`results/`) but every path can be overridden in `scripts/configs/default.yaml` or via CLI flags.

---

## 2. Dataset preparation

We use the [20BN Something-Something-V2](https://developer.qualcomm.com/software/qualcomm-ai-research-datasets/something-something-v2)
dataset. SSv2 is ~20 GB total; this project only needs **parts `00` and `01`** (~19 GB combined) because the
10 selected classes have enough clips in those two parts.

### 2.1 The 10-class subset

The 10 classes form 4 time-reversal pairs plus 2 near-static controls:

| short name   | SSv2 template                                  | pair partner   |
|--------------|------------------------------------------------|----------------|
| `opening`    | Opening [something]                            | `closing`      |
| `closing`    | Closing [something]                            | `opening`      |
| `move_up`    | Moving [something] up                          | `move_down`    |
| `move_down`  | Moving [something] down                        | `move_up`      |
| `push_L2R`   | Pushing [something] from left to right         | `push_R2L`     |
| `push_R2L`   | Pushing [something] from right to left         | `push_L2R`     |
| `put_into`   | Putting [something] into [something]           | `take_out`     |
| `take_out`   | Taking [something] out of [something]          | `put_into`     |
| `holding`    | Holding [something]                            | (static)       |
| `showing`    | Showing [something] to the camera              | (static)       |

Default split: **150 train + 40 val per class** → 1500 train / 400 val = 1900 clips total.

### 2.2 Download + decode

```bash
# Stage 1: download parts 00, 01 + the labels zip into data/raw/
python scripts/download_data.py --config scripts/configs/default.yaml

# Stage 2: pick the subset, decode 16 uniformly-sampled frames per clip (224x224, RGB),
#           run integrity checks, write data/frames.npy (~4.6 GB memmap) + results/dataset_index.csv
python scripts/prepare_dataset.py --config scripts/configs/default.yaml

# Stage 3 (optional): dataset statistics + frame strips + high/low-change visualization
python scripts/visualize_dataset.py --config scripts/configs/default.yaml
```

What Stage 2 does, in detail:

1. **Subset selection** (`src/data.py:pick_subset`) — reads `train.json` / `validation.json`, filters to the
   10 templates, shuffles with `seed=0`, takes 150 / 40 per class. *Done before extracting any video, so we
   only ever touch ~1 900 of the ~220 k SSv2 clips.*
2. **Selective extraction** (`src/data.py:extract_videos`) — writes a `members.txt` of the form
   `20bn-something-something-v2/<id>.webm` and pipes `cat parts... | tar -xzf - -T members.txt` so we read
   the 19 GB tar stream exactly once.
3. **Frame decoding** (`src/data.py:load_clip`) — OpenCV sequential decode (webm seeking is unreliable),
   uniform sampling of 16 frames with `np.linspace(0, n-1, 16).round().astype(int)`, squash-resize to
   224×224 RGB. *Squash — never centre-crop — so the moving object is never lost.*
4. **Integrity filter** (`src/data.py:integrity_ok`) — a clip is `valid=False` if it has < 16 decoded
   frames, if sampled indices are not strictly increasing, or if timestamps are non-monotonic.
   ~10 of 1 900 clips fail this and are kept in the CSV but skipped during training.
5. **Memmap write** — `np.lib.format.open_memmap("data/frames.npy", shape=(1900, 16, 224, 224, 3), dtype=uint8)`.
   Row `i` of the memmap ↔ row `i` of `results/dataset_index.csv`. **Never drop rows** — always filter by
   `valid` so indices stay aligned.
6. **Motion score** (`src/data.py:motion`) — mean absolute pixel difference between consecutive sampled
   frames, used as a per-class sanity check (the two static classes — `holding`, `showing` — should have
   low motion; reversal pairs should have similar motion).

`results/dataset_index.csv` columns:

| id      | cls       | split | text                        | path                    | n_decoded | fps  | dur_s | h    | w    | valid | motion | first_last |
|---------|-----------|-------|-----------------------------|-------------------------|-----------|------|-------|------|------|-------|--------|------------|
| 1       | opening   | train | Opening a box               | data/raw/.../1.webm     | 45        | 12   | 3.75  | 240  | 427  | True  | 17.45  | 38.2       |

---

## 3. Embedding extraction

Frame embeddings are extracted once and cached to disk; all downstream training runs on the cache, so a full
training experiment costs seconds, not hours.

### 3.1 Supported encoders

| key         | HF model id                          | dim  | notes                                          |
|-------------|--------------------------------------|------|------------------------------------------------|
| `dino`      | `facebook/dinov2-base`               | 768  | self-supervised, spatially rich (default)      |
| `clip`      | `openai/clip-vit-base-patch16`       | 512  | language-aligned, post-LN CLS token            |
| `siglip`    | `google/siglip-base-patch16-224`     | 768  | language-aligned, sigmoid loss, attention pool |
| `internvit` | `OpenGVLab/InternViT-300M-448px`     | 3072 | 448 px input, 1025 patch tokens                |

### 3.2 What gets cached

For every encoder `k` and every valid clip, three arrays are written into `embeddings/`:

| file                       | shape                       | description                                                |
|----------------------------|-----------------------------|------------------------------------------------------------|
| `<k>_cls.npy`              | `(N_clips, 16, D)`          | CLS-token (or attention-pool) global embedding per frame   |
| `<k>_patchmean.npy`        | `(N_clips, 16, D)`          | mean of patch tokens per frame                             |
| `<k>_grid2x2.npy`          | `(N_clips, 16, 4, D)`       | 2×2 spatial pooling of patch tokens (coarse layout)        |

`N_clips` is the full `len(dataset_index)` (including invalid rows) so row indices stay aligned with the
memmap and CSV; invalid rows contain zeros and are filtered at load time.

### 3.3 Run

```bash
# extract all four encoders (edit scripts/configs/default.yaml to enable/disable)
python scripts/extract_embeddings.py --config scripts/configs/default.yaml

# or just DINOv2
python scripts/extract_embeddings.py --encoders dino
```

Implementation notes (`src/embeddings.py`):

- Frames are loaded from `data/frames.npy` directly (already 224×224), normalized per encoder
  (ImageNet stats for DINOv2, CLIP mean/std for CLIP & InternViT, 0.5/0.5 for SigLIP), and stacked into
  `(B * 16, 3, 224, 224)` batches — typically 8 clips per forward = 128 frames.
- Run in `torch.float16` by default with an overflow guard: if `not torch.isfinite(g).all()`, the batch is
  re-run in fp32 and the model is switched to fp32 for the rest of the extraction.
- InternViT needs 448 px input — frames are upsampled with bicubic interpolation
  (this adds no detail, it just matches the encoder's training resolution).
- Memory is freed between encoders (`gc.collect()` + `torch.cuda.empty_cache()`).

### 3.4 Embedding analysis (optional)

`src/visualization.py:embedding_analysis` reproduces the diagnostic figures from the notebook:

- `results/fig_cosine_by_class.png` — consecutive-frame and first-vs-last cosine similarity per class,
  **rescaled** by each encoder's random-frame background similarity so 0 ≈ "as similar as two random
  frames" and 1 ≈ "identical".
- `results/fig_sequence_similarity.png` — 10×10 class similarity matrices for the *average frame* and the
  *direction of change*; reversal partners have high average-frame similarity but **negative**
  direction-of-change similarity (the arrow of time lives in the delta).
- `results/fig_trajectories_pca.png` — 2-D PCA (fit on 20 k random within-clip-centered frame deltas) of
  the embedding trajectory of each reversal pair; the start (○) and end (▶) of the class-mean path are
  marked.
- `results/fig_temporal_pooling.png` — within-class vs between-class cosine gap for single-frame / mean /
  max pooling, plus the "pooling a clip and its reverse gives the same vector" sanity check (≈ 1.0 for all
  encoders, proving that mean/max pooling is provably order-blind).

---

## 4. Training

### 4.1 Three tiers of models

The notebook compares three tiers on the same test set (400 val clips); `src/models.py` + `src/train.py`
implement tiers B and C, tier A lives in `src/evaluate.py:zeroshot_clip`.

| tier | name                          | trained? | description                                                       |
|------|-------------------------------|----------|-------------------------------------------------------------------|
| A    | `clip_zeroshot`               | no       | CLIP/SigLIP text-image cosine, averaged over frames               |
| B    | `proto_single_frame`          | no       | per-class prototype from middle frame; nearest-prototype classify |
| B    | `proto_mean_pool`             | no       | prototype from mean-pooled clip                                   |
| B    | `proto_temporal(mean+delta)`  | no       | prototype from `[mean, last_5.mean - first_5.mean]`               |
| C    | `single_frame`                | yes      | MLP on a random frame (train) / middle frame (test)              |
| C    | `mean_pool`                   | yes      | MLP on mean-pooled clip (order-blind control)                    |
| C    | `gru`                         | yes      | GRU over the 16 frame embeddings                                  |
| C    | `tcn`                         | yes      | 2-layer 1-D conv (kernel 3, dilation 1 then 2)                    |
| C    | `transformer`                 | yes      | 2-layer Transformer encoder + learned positions + [CLS]          |
| C    | `delta_transformer`           | yes      | transformer whose tokens are `[frame, frame - prev_frame]`       |

Each temporal model (C) comes with a **`_shuftrain` twin** trained on shuffled frame order — the gap
between a model and its twin on the *ordered* test set measures how much the model actually exploits
frame order vs just pooling more frames.

### 4.2 Features fed to the trained models

`src/dataset.py:load_feats(encoder)` concatenates `<encoder>_cls.npy` and the flattened `<encoder>_grid2x2.npy`
and standardizes them with **train-split statistics only** (mean / std computed on `F[tri].reshape(-1, D)`),
giving a `[N_clips, 16, D]` float32 array. DINOv2 → `D = 768 + 4·768 = 3840`.

### 4.3 Training recipe (fixed a priori, in `scripts/configs/train.yaml`)

| param        | value                    |
|--------------|--------------------------|
| seeds        | `[0, 1, 2]`              |
| epochs       | 40                       |
| batch size   | 64                       |
| optimizer    | AdamW                    |
| lr           | 1e-3                     |
| weight decay | 0.05                     |
| scheduler    | OneCycleLR, `pct_start=0.1` |
| loss         | cross-entropy, `label_smoothing=0.1` |
| grad clip    | 1.0                      |
| d_model      | 128 (GRU/TCN/Transformer)|
| dropout      | 0.2–0.3                  |

### 4.4 Run

```bash
# train every (encoder × model × seed) combination; writes models/<encoder>_<model>_seed<k>.pt
python scripts/train_models.py --config scripts/configs/default.yaml

# train just DINOv2 + delta_transformer, seeds 0..2
python scripts/train_models.py --encoders dino --models delta_transformer --seeds 0 1 2
```

Training all 3 encoders × 10 models × 3 seeds = 90 runs takes ~25 min on a T4 (every model is < 1 M params,
runs on cached 3840-dim embeddings).

---

## 5. Evaluation

### 5.1 The three test conditions

For every trained model, accuracy is computed on the 400 val clips under three frame orderings:

| condition   | frames fed to the model                         |
|-------------|-------------------------------------------------|
| `ordered`   | the original 16 uniformly-sampled frames        |
| `shuffled`  | the 16 frames, randomly permuted per clip       |
| `reversed`  | the 16 frames in reverse temporal order         |

Plus two decompositions:

- `acc_pair` — accuracy on the 320 clips belonging to a reversal pair (8 classes × 40).
- `acc_static` — accuracy on the 80 static-control clips (`holding`, `showing`).
- `flip` — of the clips the model got right under `ordered`, what fraction does it predict the **partner**
  class under `reversed`? This is the cleanest single number for "does the model read the arrow of time?"

### 5.2 Paired bootstrap

`src/evaluate.py:paired_bootstrap` does 2 000 resamples over the 400 test clips and reports
`mean accuracy difference [2.5 %, 97.5 %]` for each pair of conditions, e.g.:

```
delta_transformer − single_frame           : +0.142 [+0.098, +0.181]
delta_transformer(ordered) − its shuffle-trained twin (shuffled test) : +0.087 [+0.041, +0.131]
```

If the CI excludes 0, the difference is significant.

### 5.3 Robustness & progressive observation (`scripts/evaluate_models.py --mode robustness`)

- **Progressive observation** — re-evaluate every model feeding it only the first *k* or last *k* frames
  (k ∈ {1, 2, 3, 4, 6, 8, 12, 16}) to see how quickly each architecture locks in its prediction.
- **Corruption** — re-encode the test frames through DINOv2 after applying one of:
  - `blur` — Gaussian blur σ=6 (only large shapes survive)
  - `noise` — +0.2·N(0,1) per pixel
  - `occlusion` — gray 112×112 square at a random location, on all frames or 50 %
  - `drop` — keep 8 / 4 random frames in their original order
- **Per-clip example** — picks a pair-class clip where temporal models are right and `single_frame` is
  wrong, and renders all 6 corruption conditions side-by-side with each model's prediction.

### 5.4 Run

```bash
# main results table + paired bootstrap → results/section4_all_tiers.csv + results/fig_section4_all.png
python scripts/evaluate_models.py --config scripts/configs/default.yaml --mode main

# progressive observation + corruption → results/section5_*.csv + results/fig_section5_*.png
python scripts/evaluate_models.py --mode robustness
```

---

## 6. VLM zero-shot (`vlm/`)

`src/vlm.py` runs two open-weights generative VLMs as **multiple-choice classifiers** on the same 400 val
clips (subsampled to 10 per class = 100 clips by default to keep runtime reasonable):

| name             | HF model id                                  | params |
|------------------|----------------------------------------------|--------|
| `Qwen2-VL-2B`    | `Qwen/Qwen2-VL-2B-Instruct`                  | 2 B    |
| `SmolVLM2-500M`  | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 500 M  |

### 6.1 Prompt

Each VLM sees 1, 8 (ordered), or 8 (reversed) frames and is asked:

> *These are N frames from a short first-person video of a hand interacting with an object, in
> chronological order (Frame 1 is earliest). Left/right are as seen in the images. … Which action is
> shown? A. Opening something … J. Showing something to the camera. Answer with the letter only.*

The answer distribution is recovered by softmaxing the next-token logits over the 10 letter tokens
(both `"A"` and `" A"`), averaged over two random option orderings to wash out position bias.

### 6.2 Conditions

- `single` — only the middle frame
- `ordered` — 8 evenly-spaced frames in chronological order
- `reversed` — the same 8 frames in reverse

Plus a free-form `describe` call on 6 qualitative clips asking *"what changes between the first and last
frame, and what becomes clear only over time?"* under both `ordered` and `reversed` — the answers are
saved to `vlm/qualitative_<model>.json`.

### 6.3 Run

```bash
python scripts/run_vlm_zeroshot.py --config scripts/configs/default.yaml --models Qwen2-VL-2B SmolVLM2-500M
```

Output: `results/vlm_results.csv` (acc / acc_pair / acc_static / reversal-flip-rate per model × condition)
and `results/vlm_vs_embedding.png` comparing VLMs to the embedding-based models on the same 100 clips.

### 6.4 Patching `num2words`

`transformers` < 4.46 imports `num2words` lazily inside SmolVLM's processor; the script monkey-patches
`transformers.models.smolvlm.processing_smolvlm.num2words` before loading the model to avoid the
`ImportError` on minimal environments. See `src/vlm.py:patch_num2words`.

---

## 7. Reproduction of your results

End-to-end, from a fresh clone to the full results table:

```bash
# 0. env
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1. data (~30 min on a fast connection; ~10 min to decode frames)
python scripts/download_data.py
python scripts/prepare_dataset.py
python scripts/visualize_dataset.py           # → results/dataset_stats.png, frame_strips_per_class.png

# 2. embeddings (~30 min on a T4 for all 4 encoders)
python scripts/extract_embeddings.py

# 3. train (~25 min for 3 encoders × 10 models × 3 seeds)
python scripts/train_models.py

# 4. evaluate (~5 min)
python scripts/evaluate_models.py --mode main
python scripts/evaluate_models.py --mode robustness

# 5. VLM zero-shot (~45 min on a T4 for both models on 100 clips)
python scripts/run_vlm_zeroshot.py
```

**Reproducibility guarantees.**

- All randomness (Python `random`, NumPy, PyTorch, CUDA) is seeded via `src/utils.py:set_seed`.
  Seed is `0` by default for data prep and `[0, 1, 2]` for training.
- The data subset is deterministic: `random.Random(0).shuffle(ids)` per class, so the same 1 900 clips
  are picked on every machine.
- Feature standardization uses **train-split statistics only** — no leakage from the val split.
- Every experiment reads the same `scripts/configs/default.yaml`; override any field with
  `--config path/to/override.yaml` or `--set field.subfield=value`.
- Results CSVs and figures are committed to `results/`; the multi-GB raw arrays in `data/`,
  `embeddings/`, `models/` are gitignored but can be regenerated by the commands above.

**To reproduce a single figure or table**, every results file has a one-line command in its header
comment; e.g. `results/section4_all_tiers.csv` is produced by
`python scripts/evaluate_models.py --mode main`.

---

## Citation

If you use this code, please cite the SSv2 dataset:

```bibtex
@inproceedings{goyal2017something,
  title={The "something something" video database for learning and evaluating visual common sense},
  author={Goyal, Raghav and Kahou, Samira Ebrahimi and Michalski, Vincent and others},
  booktitle={ICCV},
  year={2017}
}
```

and the backbone papers (DINOv2, CLIP, SigLIP, InternViT) as appropriate.

## License

MIT — see [LICENSE](LICENSE). SSv2 itself has its own research-use license from Qualcomm / Twenty Billion Neurons.
