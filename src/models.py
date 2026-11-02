"""Model zoo (README §4.1).

Every model takes a ``(B, T, D)`` tensor of frozen frame embeddings and produces
``(B, C)`` logits. All models are < 1 M parameters — they run on cached
embeddings, so training is seconds-per-epoch.

Two flavours:

* **Order-blind baselines** — ``SingleFrame`` (one frame → MLP) and ``MeanPool``
  (mean over time → MLP). These set the floor for "how much can you get without
  reading frame order?".
* **Temporal models** — ``GRU``, ``TCN``, ``TemporalTransformer``,
  ``DeltaTransformer``. Each has a ``*_shuftrain`` variant whose class is the
  same but whose training data is frame-shuffled every batch; the gap between a
  model and its shuffle-trained twin measures how much the model actually
  exploits order vs just pools more frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Order-blind baselines
# ---------------------------------------------------------------------------

class SingleFrame(nn.Module):
    """One frame → MLP. Random frame during training, middle frame at test time."""

    def __init__(self, d_in: int, hidden: int = 256, n_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if self.training:
            t = torch.randint(0, x.shape[1], (len(x),), device=x.device)
            x_sel = x[torch.arange(len(x), device=x.device), t]
        else:
            x_sel = x[:, x.shape[1] // 2]
        return self.net(x_sel)


class MeanPool(nn.Module):
    """Mean over time → MLP. Provably order-blind (proven in §3 of the README)."""

    def __init__(self, d_in: int, hidden: int = 256, n_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(1))


# ---------------------------------------------------------------------------
# Temporal models
# ---------------------------------------------------------------------------

class GRUNet(nn.Module):
    """GRU over the 16 frame embeddings → MLP head on the final hidden state."""

    def __init__(self, d_in: int, d_model: int = 128, n_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_model), nn.GELU())
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(self.proj(x))
        return self.head(h[-1])


class TCN(nn.Module):
    """Two-layer 1-D temporal conv (kernel 3, dilation 1 then 2) → mean-pool → MLP."""

    def __init__(self, d_in: int, d_model: int = 128, n_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_model))
        self.c1 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.c2 = nn.Conv1d(d_model, d_model, 3, padding=2, dilation=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).transpose(1, 2)  # (B, d_model, T)
        h = F.gelu(self.c1(h))
        h = F.gelu(self.c2(h))
        return self.head(h.mean(2))


class TemporalTransformer(nn.Module):
    """2-layer Transformer encoder + learned positions + [CLS] token.

    The standard recipe for sequence classification with a frozen embedding
    backbone. ``DeltaTransformer`` extends this by concatenating each frame's
    embedding with its delta from the previous frame.
    """

    def __init__(
        self,
        d_in: int,
        n_frames: int = 16,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_frames, d_model) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 2 * d_model, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        tokens = self.proj(x) + self.pos[:, :x.shape[1]]
        cls = self.cls.expand(len(x), -1, -1)
        h = self.enc(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0])


class DeltaTransformer(TemporalTransformer):
    """TemporalTransformer whose tokens are ``[frame, frame - prev_frame]``.

    Makes the direction of change a first-class input — designed to make the
    arrow of time easy to read for the model.
    """

    def __init__(self, d_in: int, **kwargs) -> None:
        super().__init__(2 * d_in, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
        return super().forward(torch.cat([x, delta], dim=-1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: name → (class, train_on_shuffled_order?)
MODELS: dict[str, tuple[type[nn.Module], bool]] = {
    "single_frame":               (SingleFrame,         False),
    "mean_pool":                  (MeanPool,            False),
    "gru":                        (GRUNet,              False),
    "tcn":                        (TCN,                 False),
    "transformer":                (TemporalTransformer, False),
    "delta_transformer":          (DeltaTransformer,    False),
    "gru_shuftrain":              (GRUNet,              True),
    "tcn_shuftrain":              (TCN,                 True),
    "transformer_shuftrain":      (TemporalTransformer, True),
    "delta_transformer_shuftrain": (DeltaTransformer,   True),
}


def build_model(name: str, d_in: int, *, d_model: int = 128, dropout: float = 0.3, n_classes: int = 10) -> nn.Module:
    """Instantiate a model by registry name."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choices: {list(MODELS)}")
    cls, _ = MODELS[name]
    # SingleFrame / MeanPool use `hidden=256` + `dropout=0.3`; temporal models
    # use `d_model=128` + `dropout=0.2`. Match the notebook's defaults.
    if cls in (SingleFrame, MeanPool):
        return cls(d_in, hidden=256, n_classes=n_classes, dropout=dropout if dropout is not None else 0.3)
    return cls(d_in, d_model=d_model, n_classes=n_classes, dropout=0.2 if dropout is None else dropout)


def trains_on_shuffled(name: str) -> bool:
    """Whether ``name``'s training loop should feed frame-shuffled batches."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}")
    return MODELS[name][1]


def count_parameters(m: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
