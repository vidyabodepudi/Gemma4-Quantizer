"""
gemma4-quantizer: Native 3D tensor quantization for Google Gemma 4 MoE models.

Instead of unfusing expert weights into individual nn.Linear layers (wasteful),
this library quantizes fused 3D expert tensors [num_experts, intermediate, hidden]
directly with per-expert-channel granularity.
"""

__version__ = "0.1.0"

from gemma4_quant.detector import FusedExpertDetector
from gemma4_quant.quantizer import (
    Gemma4Quantizer,
    QuantConfig,
    QuantMethod,
)

__all__ = [
    "FusedExpertDetector",
    "Gemma4Quantizer",
    "QuantConfig",
    "QuantMethod",
]
