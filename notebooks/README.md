# `notebooks/` — original exploratory Colab notebook

This folder holds the original Colab notebook that this repo was refactored
from. It's kept for reference / reproducibility — the canonical, tested code
lives in `src/` and `scripts/`.

## File

| file                    | description                                                        |
|-------------------------|--------------------------------------------------------------------|
| `Untitled14.ipynb`      | the original Colab notebook (40 cells, ~60 KB)                     |

## What's in the notebook

The notebook has 6 sections, each mapping 1-to-1 to a stage of this repo:

| notebook section                  | cells   | repo stage                                              |
|-----------------------------------|---------|---------------------------------------------------------|
| 1. Data preparation               | 0-12    | `scripts/download_data.py` + `prepare_dataset.py` + `visualize_dataset.py` |
| 2. Embedding extraction           | 14-17   | `scripts/extract_embeddings.py`                         |
| 3. Embedding analysis             | 18-23   | `src/visualization.py:embedding_analysis`               |
| 4. Single vs temporal models      | 24-28   | `scripts/train_models.py` + `evaluate_models.py --mode main` |
| 5. Robustness + progressive obs.  | 29-33   | `scripts/evaluate_models.py --mode robustness`          |
| 6. VLM zero-shot                  | 35-37   | `scripts/run_vlm_zeroshot.py`                           |

## Why keep the notebook?

- **Reproducibility** — the notebook is the exact artifact that produced the
  results in `results/`. If a reviewer wants to see the *exact* sequence of
  cells that ran, they can read it here.
- **Pedagogy** — the notebook's interleaved markdown + code is easier to
  follow than the modular `src/` layout for someone reading the project
  top-to-bottom.
- **Cell-level diffs** — when iterating on the research, it's often faster
  to edit a cell and re-run it than to change `src/` + rerun the script.

## Why not run the notebook?

- It hardcodes Colab paths (`/content/drive/MyDrive/ssv2`, `/content/cache`).
- It calls `drive.mount()` which only works inside Colab.
- It uses global-state cells (e.g. cell 25 reads `vidx` from cell 15) —
  fragile to out-of-order execution.
- It has no CLI, no config, no tests.

The `src/` + `scripts/` layout fixes all of the above. Run the scripts; read
the notebook for context.
