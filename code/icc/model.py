"""Official-component composition and the proposed three-stage extension."""
from dataclasses import dataclass, asdict

import torch


MODALITIES = ("T", "V", "A")
COMBINATIONS = {"T": (1, 0, 0), "V": (0, 1, 0), "A": (0, 0, 1),
                "TV": (1, 1, 0), "TA": (1, 0, 1), "VA": (0, 1, 1), "TVA": (1, 1, 1)}


@dataclass
class ModelConfig:
    input_dims: tuple = (768, 512, 768)
    hidden: int = 256
    heads: int = 4
    topics: int = 8
    conditional: bool = True
    compensation: bool = True
    a_max: float = 0.5
    gain_scale: float = 1.0
    mixing: str = "gain"
    backbone: str = "lightweight"
    sequence_lengths: tuple = (1, 1, 1)
    layers: int = 2
    memory_layers: int = 3
    encoder_layers: int = 1
    dropout: float = 0.1
    prompt_length: int = 8

    def __post_init__(self):
        if len(self.input_dims) != 3 or min(self.input_dims) < 1:
            raise ValueError("Expected three positive T,V,A dimensions")
        if self.hidden % self.heads or self.hidden < 4 or self.topics < 2:
            raise ValueError("Invalid attention/topic dimensions")
        if not 0 <= self.a_max <= 1 or self.gain_scale <= 0:
            raise ValueError("a_max must be in [0,1]; gain_scale must be positive")
        if self.mixing not in {"gain", "direct", "constant"}:
            raise ValueError("Unknown mixing rule")
        if self.backbone not in {"lightweight", "concat", "mult", "mplmm"}:
            raise ValueError("Unknown backbone")
        if len(self.sequence_lengths) != 3 or min(self.sequence_lengths) < 1:
            raise ValueError("Expected three positive sequence lengths in T,V,A order")
        if min(self.layers, self.memory_layers, self.encoder_layers, self.prompt_length) < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid backbone depth/dropout/prompt length")
        if self.backbone == "mplmm" and self.compensation:
            raise ValueError("Standalone MPLMM already includes completion; use compensation=False")


def validate_mask(mask, batch=None):
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != 3:
        raise ValueError("Mask must be bool [batch,3] in T,V,A order")
    if batch is not None and mask.shape[0] != batch:
        raise ValueError("Mask batch mismatch")
    if not mask.any(1).all():
        raise ValueError("All-missing input is outside the seven-combination protocol")


def sanitize(features, mask, dims):
    """Mask BEFORE any learned operation. Even hidden NaN/Inf cannot contaminate."""
    validate_mask(mask)
    clean = []
    for k, (x, d) in enumerate(zip(features, dims)):
        if x.ndim not in (2, 3) or x.shape[0] != len(mask) or x.shape[-1] != d or not x.is_floating_point():
            raise ValueError(f"{MODALITIES[k]} must be floating [batch,optional_length,{d}]")
        x = torch.where(mask[:, k].reshape(-1, *([1]*(x.ndim-1))), x, torch.zeros_like(x))
        if not torch.isfinite(x).all():
            raise ValueError("Visible features contain NaN/Inf")
        clean.append(x)
    if len(features) != 3:
        raise ValueError("Expected exactly T,V,A features")
    return tuple(clean)


def mask_codes(mask):
    return (mask.long() * torch.tensor([4, 2, 1], device=mask.device)).sum(1)


