# `evaluation/` — evaluation reports and methodology notes

This folder documents **how** the project evaluates models. The actual
result CSVs and figures are written to `results/`; this folder holds
higher-level write-ups and any analysis notebooks/scripts that don't fit
elsewhere.

## What lives here

| file                              | what it is                                                |
|-----------------------------------|-----------------------------------------------------------|
| `README.md` (this file)           | the evaluation methodology                                |
| `methodology.md` (optional)       | long-form write-up of the three test conditions + bootstrap |
| `confusion_matrices/` (optional)  | per-(encoder, model) 10×10 confusion matrix PNGs           |

## The three test conditions

Every trained model is evaluated on the 400 val clips under three frame
orderings:

### `ordered` — the original 16 uniformly-sampled frames, in chronological order

This is the headline number. A model that can't beat `0.10` (chance for 10
classes) is broken; a model that hits `0.7+` on the reversal pairs is reading
the arrow of time correctly.

### `shuffled` — the 16 frames, randomly permuted per clip

A model that **uses** frame order should drop noticeably under shuffling.
A model whose `shuffled` accuracy equals its `ordered` accuracy is provably
order-blind — either by construction (`MeanPool`, `proto_mean_pool`) or
because it learned to ignore order.

### `reversed` — the 16 frames in reverse temporal order

The cleanest single test for "does the model read the arrow of time?". On
reversal-pair classes, a model that understands direction should predict the
**partner** class under reversal:

| ground truth | ordered prediction | reversed prediction (good model) |
|--------------|--------------------|-----------------------------------|
| opening      | opening            | closing                           |
| push_L2R     | push_L2R           | push_R2L                          |
| put_into     | put_into           | take_out                          |

The **`flip`** metric = of the clips the model got right under `ordered`,
what fraction does it predict the partner class under `reversed`? A flip
rate near 1.0 means the model has a perfect sense of direction; near 0.0
means it's reading something else (e.g. object identity, not motion).

## The two accuracy decompositions

- **`acc_pair`** — accuracy on the 320 clips belonging to a reversal pair
  (8 classes × 40 val). This is the hard half — you can't get these right
  without reading frame order.
- **`acc_static`** — accuracy on the 80 static-control clips (`holding`,
  `showing`). These don't have a direction, so a single frame suffices; this
  number is the "ceiling for order-blind models".

## Paired bootstrap

`src/evaluate.py:paired_bootstrap` resamples the 400 test clips 2 000 times
with replacement and reports the 95 % confidence interval for the difference
in accuracy between any two conditions. Example:

```
delta_transformer − single_frame           : +0.142 [+0.098, +0.181]
delta_transformer(ordered) − its shuffle-trained twin (shuffled test) : +0.087 [+0.041, +0.131]
```

If the CI excludes 0, the difference is significant at p < 0.05.

## Robustness

`scripts/evaluate_models.py --mode robustness` re-runs every model under:

1. **Progressive observation** — feed only the first / last / evenly-sampled
   `k` frames (k ∈ {1, 2, 3, 4, 6, 8, 12, 16}).
2. **Corruption** — re-encode the test frames through DINOv2 after applying
   one of: Gaussian blur (σ=6), heavy noise (+0.2·N(0,1)), 112×112 gray
   occlusion square (on all frames or 50 %), or random frame dropping
   (keep 8 / 4 of 16).

These experiments isolate *what* each architecture relies on: temporal models
should degrade more gracefully under frame dropping (they get the same
information from fewer frames) but more sharply under heavy blur (they lose
the high-frequency cues that disambiguate direction).

## Reproducing a single number

Every CSV in `results/` has a one-line command in its header comment. For
example, to reproduce just the main results table:

```bash
python scripts/evaluate_models.py --mode main
# → results/section4_all_tiers.csv
```
