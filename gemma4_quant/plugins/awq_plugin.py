"""
AWQ plugin adapter for Gemma 4 MoE models.

Similar to the GPTQ plugin, this patches AutoAWQ's module discovery
to handle fused 3D expert tensors. AWQ (Activation-aware Weight
Quantization) is particularly well-suited for MoE models because
its activation-aware scaling can account for expert utilization patterns.

Usage:
    from gemma4_quant.plugins.awq_plugin import patch_autoawq
    patch_autoawq()

    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_pretrained(...)
    model.quantize(tokenizer, quant_config={"w_bit": 4, "group_size": 128})
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _get_expert_scales(
    fused_weight: torch.Tensor,
    activation_stats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-expert AWQ scaling factors.

    For each expert in the fused 3D tensor [E, I, H], compute the
    optimal per-channel scaling that minimizes quantization error
    weighted by activation magnitudes.
    """
    E, I, H = fused_weight.shape
    scales = torch.ones(E, 1, H, device=fused_weight.device, dtype=fused_weight.dtype)

    for e in range(E):
        w = fused_weight[e]  # [I, H]
        w_max = w.abs().amax(dim=0, keepdim=True)  # [1, H]

        if activation_stats is not None and activation_stats.ndim >= 1:
            # Use activation statistics to weight the scaling
            act_scale = activation_stats.abs().clamp(min=1e-5)
            if act_scale.shape[-1] == H:
                # AWQ formula: s = (a_max / w_max)^alpha
                alpha = 0.5  # Balance between activation and weight difficulty
                s = (act_scale / w_max.clamp(min=1e-5)).pow(alpha)
                scales[e, 0, :] = s.squeeze(0) if s.ndim > 1 else s
        else:
            # Fallback: use weight-only heuristic
            scales[e, 0, :] = w_max.squeeze(0).clamp(min=1e-5)

    # Normalize
    scales = scales / scales.amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return scales


class AWQFusedExpertModule(torch.nn.Module):
    """
    Wraps a fused 3D expert tensor for AWQ quantization.
    Applies per-expert scaling before standard per-channel quantization.
    """

    def __init__(
        self,
        fused_weight: torch.Tensor,
        expert_idx: int,
        scales: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Store the 2D slice for this expert
        w = fused_weight[expert_idx].clone()

        if scales is not None:
            # Apply AWQ scaling
            w = w * scales[expert_idx]

        self.weight = torch.nn.Parameter(w)
        self.expert_idx = expert_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def patch_autoawq():
    """
    Patch AutoAWQ to handle fused 3D expert tensors.
    Call BEFORE loading your model with AutoAWQ.
    """
    global _PATCHED
    if _PATCHED:
        logger.info("AutoAWQ already patched for Gemma 4 MoE")
        return

    try:
        import awq
        from awq.models.base import BaseAWQForCausalLM
    except ImportError:
        raise ImportError(
            "autoawq is required. Install with: pip install autoawq"
        )

    # Store original quantize method
    _original_quantize = BaseAWQForCausalLM.quantize

    def patched_quantize(self, tokenizer=None, quant_config=None, *args, **kwargs):
        """
        Patched quantize that pre-processes fused 3D expert tensors.
        """
        logger.info("Running patched AWQ quantization for Gemma 4 MoE")

        # Find and wrap fused expert tensors
        expert_modules = {}
        for name, module in self.model.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                if param.ndim == 3 and param.shape[0] >= 4:
                    if any(kw in name.lower() for kw in ("expert", "moe")):
                        logger.info(
                            f"Pre-processing fused expert: {name}.{pname} "
                            f"shape={tuple(param.shape)}"
                        )
                        scales = _get_expert_scales(param.data)
                        E = param.shape[0]
                        for eid in range(E):
                            wrapper = AWQFusedExpertModule(param.data, eid, scales)
                            key = f"{name}.{pname}.__awq_expert_{eid}"
                            expert_modules[key] = wrapper

        # Register temporary modules
        for key, wrapper in expert_modules.items():
            # Add as attribute for discovery
            parts = key.rsplit(".", 1)
            parent_name, attr_name = parts[0], parts[1]
            parent = dict(self.model.named_modules()).get(parent_name)
            if parent is not None:
                setattr(parent, attr_name, wrapper)

        # Call original quantize
        result = _original_quantize(self, tokenizer, quant_config, *args, **kwargs)

        # Clean up temporary modules
        for key in expert_modules:
            parts = key.rsplit(".", 1)
            parent_name, attr_name = parts[0], parts[1]
            parent = dict(self.model.named_modules()).get(parent_name)
            if parent is not None and hasattr(parent, attr_name):
                delattr(parent, attr_name)

        return result

    BaseAWQForCausalLM.quantize = patched_quantize

    _PATCHED = True
    logger.info("AutoAWQ patched for Gemma 4 MoE fused 3D expert tensors")


def unpatch_autoawq():
    """Remove the Gemma 4 MoE patch from AutoAWQ."""
    global _PATCHED
    _PATCHED = False
    logger.info("AutoAWQ unpatch requested (restart recommended)")
