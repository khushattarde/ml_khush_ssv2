"""Configuration dataclass loaded from YAML.

Single source of truth for every hyper-parameter in the pipeline. The default
config lives at ``scripts/configs/default.yaml``; any field can be overridden
by passing a different YAML file or by ``--set path.to.field=value`` on the CLI.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils import repo_root


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset preparation (Section 1-2 of the README)."""
    raw_dir: str = "data/raw"                # downloaded tar parts + labels
    labels_pkg_dir: str = "data/labels_pkg"  # extracted train.json / validation.json
    frames_path: str = "data/frames.npy"     # memmap of decoded frames
    index_csv: str = "results/dataset_index.csv"
    parts: tuple[str, ...] = ("00", "01")    # SSv2 tar parts to download
    n_frames: int = 16
    img_size: int = 224
    max_train_per_class: int = 150
    max_val_per_class: int = 40
    seed: int = 0
    # URLs — Qualcomm mirrors; re-host if they move.
    video_url_tmpl: str = (
        "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/"
        "dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-{part}"
    )
    labels_url: str = (
        "https://softwarecenter.qualcomm.com/api/download/software/dataset/"
        "AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"
    )


@dataclass
class EmbeddingConfig:
    """Embedding extraction (Section 3)."""
    cache_dir: str = "embeddings"
    encoders: tuple[str, ...] = ("dino", "clip", "siglip", "internvit")
    clips_per_batch: int = 8                  # 8 clips × 16 frames = 128 frames per forward
    internvit_clips_per_batch: int = 4        # 448 px → 1025 tokens/frame, lower batch
    internvit_resolution: int = 448
    use_fp16: bool = True
    overflow_guard: bool = True               # re-run a batch in fp32 if fp16 gave non-finite values


@dataclass
class TrainConfig:
    """Training (Section 4)."""
    checkpoints_dir: str = "models"
    encoders: tuple[str, ...] = ("dino", "clip", "siglip")
    models: tuple[str, ...] = (
        "single_frame", "mean_pool",
        "gru", "tcn", "transformer", "delta_transformer",
        "gru_shuftrain", "tcn_shuftrain",
        "transformer_shuftrain", "delta_transformer_shuftrain",
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    d_model: int = 128
    dropout: float = 0.3
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    pct_start: float = 0.1                    # OneCycleLR warmup fraction
    num_workers: int = 0                      # embeddings are tiny; no need for workers


@dataclass
class EvalConfig:
    """Evaluation (Section 5)."""
    results_dir: str = "results"
    bootstrap_samples: int = 2000
    robustness_ks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    corruption_fracs: tuple[float, ...] = (1.0, 0.5)


@dataclass
class VLMConfig:
    """VLM zero-shot (Section 6)."""
    output_dir: str = "vlm"
    models: dict[str, str] = field(default_factory=lambda: {
        "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
        "SmolVLM2-500M": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    })
    n_per_class: int = 10
    n_qualitative: int = 6
    n_frames_per_clip: int = 8                # frames shown to the VLM (sampled evenly)
    seed: int = 0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level config. ``root`` is the repo root used to resolve relative paths."""
    root: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    # ----- path resolution -----

    def resolve(self, path: str) -> Path:
        """Resolve ``path``: absolute paths are kept, relative paths are rooted at ``self.root``."""
        p = Path(path)
        return p if p.is_absolute() else (Path(self.root) / p).resolve()

    # ----- constructors -----

    @classmethod
    def default(cls) -> "Config":
        """Return a Config rooted at the repo root with all defaults."""
        cfg = cls()
        cfg.root = str(repo_root())
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a YAML config file; missing fields fall back to defaults."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls.default()
        for section_name in ("data", "embeddings", "train", "eval", "vlm"):
            section_raw = raw.get(section_name, {})
            if not isinstance(section_raw, dict):
                continue
            section = getattr(cfg, section_name)
            for k, v in section_raw.items():
                if hasattr(section, k):
                    # tuple fields come back as lists from YAML — coerce
                    cur = getattr(section, k)
                    if isinstance(cur, tuple) and isinstance(v, list):
                        v = tuple(v)
                    if isinstance(cur, dict) and isinstance(v, dict):
                        v = {**cur, **v}
                    setattr(section, k, v)
        if "root" in raw:
            cfg.root = raw["root"]
        return cfg

    # ----- CLI override -----

    def apply_overrides(self, overrides: list[tuple[str, Any]]) -> None:
        """Apply dotted-path overrides, e.g. ``[("train.epochs", 60)]``."""
        for dotted, value in overrides:
            obj: Any = self
            parts = dotted.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            cur = getattr(obj, parts[-1])
            if isinstance(cur, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, parts[-1], value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dict (useful for logging)."""
        def _ser(x: Any) -> Any:
            if dataclasses.is_dataclass(x):
                return {k: _ser(v) for k, v in dataclasses.asdict(x).items()}
            return x
        return _ser(self)


def load_config(path: str | Path | None = None, overrides: list[tuple[str, Any]] | None = None) -> Config:
    """Convenience: load from YAML (or default if None) and apply CLI overrides."""
    cfg = Config.from_yaml(path) if path else Config.default()
    if overrides:
        cfg.apply_overrides(overrides)
    return cfg
