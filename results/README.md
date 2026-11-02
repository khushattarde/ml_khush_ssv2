# `results/` — CSVs and PNGs produced by every stage

This folder is the **only** large-output folder that's committed to git (the
CSVs are small; the PNGs are a few hundred KB each). The multi-GB raw arrays
in `data/`, `embeddings/`, `models/` are gitignored.

## Files after a full pipeline run

### Data prep (Stage 3)

| file                          | producer                     | description                                |
|-------------------------------|------------------------------|--------------------------------------------|
| `dataset_index.csv`           | `prepare_dataset.py`         | per-clip metadata + integrity + motion     |
| `dataset_stats.png`           | `visualize_dataset.py`       | 4-panel summary                            |
| `frame_strips_per_class.png`  | `visualize_dataset.py`       | 1 row × 8 frames per class                 |
| `high_vs_low_change.png`      | `visualize_dataset.py`       | top-4 vs bottom-4 clips by pixel change    |

### Embedding analysis (Stage 4, with `--analyze`)

| file                                    | description                                                |
|-----------------------------------------|------------------------------------------------------------|
| `cosine_consecutive_rescaled.csv`       | consecutive-frame cosine per (encoder, class)              |
| `cosine_first_last_rescaled.csv`        | first-vs-last-frame cosine per (encoder, class)            |
| `fig_cosine_by_class.png`               | bar chart of the two above                                 |
| `sequence_similarity_summary.csv`       | reversal-partner vs other-class similarity                 |
| `fig_sequence_similarity.png`           | 10×10 class similarity matrices (avg-frame + direction)    |
| `fig_trajectories_pca.png`              | 2-D PCA of within-clip deltas, one panel per reversal pair |
| `pooling_summary.csv`                   | within vs between class cosine for 3 pooling variants      |
| `fig_temporal_pooling.png`              | bar chart of pooling gap + the order-blindness test        |

### Training + evaluation (Stages 5-6)

| file                                    | description                                                |
|-----------------------------------------|------------------------------------------------------------|
| `section4_all_tiers.csv`                | main results table: every (encoder, model) × 3 conditions |
| `fig_section4_all.png`                  | bar chart of ordered accuracy                              |
| `section4_paired_bootstrap.csv`         | paired-bootstrap CIs for 6 key comparisons per encoder     |
| `section5_progressive.csv`              | accuracy vs k frames observed                              |
| `fig_section5_progressive.png`          | line chart of the above                                    |
| `section5_corruption.csv`               | accuracy under 6 corruption conditions                     |
| `fig_section5_corruption.png`           | bar chart of the above                                     |

### VLM zero-shot (Stage 7)

| file                                    | description                                                |
|-----------------------------------------|------------------------------------------------------------|
| `vlm_results.csv`                       | per-(model, condition) accuracy + flip rate                |
| `vlm_scores_<model>.npz` *(gitignored)* | raw probability vectors (for resume)                       |
| `vlm_vs_embedding.png`                  | bar chart comparing VLMs to embedding models on 100 clips  |

## How to regenerate any single file

Every CSV has a one-line command in its header comment. Quick reference:

```bash
python scripts/visualize_dataset.py          # → dataset_stats.png, frame_strips_per_class.png, high_vs_low_change.png
python scripts/extract_embeddings.py --analyze  # → cosine_*.csv, fig_cosine_by_class.png, ...
python scripts/evaluate_models.py --mode main       # → section4_all_tiers.csv, fig_section4_all.png
python scripts/evaluate_models.py --mode robustness # → section5_*.csv, fig_section5_*.png
python scripts/run_vlm_zeroshot.py           # → vlm_results.csv, vlm_vs_embedding.png
```

## What's NOT committed here

- `vlm_scores_*.npz` — raw probability dumps (~50 KB each, gitignored because
  they're regenerable and clutter PRs).
- `vlm_qualitative_*.json` — lives in `vlm/` (it's a VLM-specific output).
- Any per-clip prediction array — these are intermediate; the aggregated CSVs
  are the source of truth.
