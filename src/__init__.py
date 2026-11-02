"""SSv2 action recognition — temporal order on frozen image embeddings.

Public API: see ``src/__init__.py``. Submodules:

* :mod:`src.config`      — dataclass config loaded from YAML
* :mod:`src.data`        — SSv2 download, subset selection, frame decoding
* :mod:`src.dataset`     — torch Dataset over cached embeddings
* :mod:`src.embeddings`  — DINOv2 / CLIP / SigLIP / InternViT extraction
* :mod:`src.models`      — SingleFrame, MeanPool, GRU, TCN, Transformer, DeltaTransformer
* :mod:`src.train`       — AdamW + OneCycleLR training loop
* :mod:`src.evaluate`    — ordered/shuffled/reversed eval, paired bootstrap, zero-shot
* :mod:`src.vlm`         — Qwen2-VL / SmolVLM2 multiple-choice zero-shot
* :mod:`src.visualization` — dataset stats, frame strips, PCA trajectories
* :mod:`src.utils`       — seeding, logging, metrics helpers
"""

from src.config import Config, load_config
from src.utils import set_seed, get_logger, partner_of

__version__ = "1.0.0"

__all__ = ["Config", "load_config", "set_seed", "get_logger", "partner_of", "__version__"]
