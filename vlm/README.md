# `vlm/` — generative VLM zero-shot classification

This folder holds the VLM-specific outputs: qualitative descriptions,
per-clip probability dumps, and the prompt template. The headline results
table lives in `results/vlm_results.csv` (see `evaluation/README.md`).

## What lives here

| file                              | what it is                                                |
|-----------------------------------|-----------------------------------------------------------|
| `README.md` (this file)           | VLM methodology + prompt template                         |
| `qualitative_<model>.json`        | free-form descriptions for 6 example clips (ordered + reversed) |
| `prompts.md` (optional)           | long-form prompt engineering notes                        |

Score dumps (`vlm_scores_<model>.npz`) are written to `results/` (gitignored)
so the run can be resumed after a crash.

## The two VLMs

| name             | HF model id                                  | params | notes                              |
|------------------|----------------------------------------------|--------|------------------------------------|
| `Qwen2-VL-2B`    | `Qwen/Qwen2-VL-2B-Instruct`                  | 2 B    | strong, ~12 GB VRAM in bf16        |
| `SmolVLM2-500M`  | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 500 M  | fits comfortably on a T4           |

## The prompt

Each VLM sees `N` frames (1, 8 ordered, or 8 reversed) and is asked:

```
These are N frame(s) from a short first-person video of a hand
interacting with an object, in chronological order (Frame 1 is earliest).
Left/right are as seen in the images.

Frame 1: <image>
Frame 2: <image>
...
Frame N: <image>

Which action is shown?
A. Opening something
B. Closing something
...
J. Showing something to the camera
Answer with the letter only.
```

### Recovering a probability distribution

The VLM isn't asked to generate text — we read the **next-token logits** after
the prompt and softmax over the 10 letter tokens (both `"A"` and `" A"` spellings,
log-sum-exp'd together). This gives a proper probability vector over the 10
classes without any sampling.

### Washing out position bias

LLMs are biased toward certain letter positions (early letters get higher
probability regardless of content). We run each clip through **two random
option orderings** (forward and reverse) and average the resulting
distributions, which cancels the position bias.

## The three conditions

| condition   | frames shown to the VLM                              |
|-------------|------------------------------------------------------|
| `single`    | only the middle frame                                |
| `ordered`   | 8 evenly-spaced frames, chronological order          |
| `reversed`  | the same 8 frames, reverse chronological order       |

The `single → ordered` gap measures how much the VLM benefits from seeing
motion. The `ordered → reversed` flip rate measures whether the VLM reads
the arrow of time the same way the trained temporal models do.

## Qualitative descriptions

For 6 example clips (one per reversal-pair class, capped at 6), the VLM is
also asked a free-form question under both `ordered` and `reversed`:

> *In 2 short sentences: what changes between the first and last frame, and
> what becomes clear only over time?*

These answers are saved to `vlm/qualitative_<model>.json` and are useful for
sanity-checking that the VLM's text reasoning aligns with its classification.

## `num2words` patch

`transformers` < 4.46 imports `num2words` lazily inside SmolVLM's processor.
`src/vlm.py:patch_num2words` monkey-patches it before loading the model. The
patch is a no-op if `num2words` is already installed (it's in
`requirements.txt`).

## Reproducing

```bash
python scripts/run_vlm_zeroshot.py                              # both VLMs, 100 clips
python scripts/run_vlm_zeroshot.py --models SmolVLM2-500M       # just SmolVLM2
```

The script is **resumable** — if `results/vlm_scores_<model>.npz` and
`vlm/qualitative_<model>.json` both exist, the model is skipped. Delete those
files to force a re-run.
