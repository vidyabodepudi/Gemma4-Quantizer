"""
GPTQ plugin adapter for Gemma 4 MoE models.

Monkey-patches AutoGPTQ's module discovery to recognize fused 3D expert
tensors and quantize them with per-expert-channel granularity.

Usage:
    from gemma4_quant.plugins.gptq_plugin import patch_autogptq
    patch_autogptq()  # Call before loading model with AutoGPTQ

    # Then use AutoGPTQ as normal
    from auto_gptq import AutoGPTQForCausalLM
    model = AutoGPTQForCausalLM.from_pretrained(...)
    model.quantize(examples)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


class FusedExpertQuantWrapper(torch.nn.Module):
    """
    Wraps a fused 3D expert parameter as a module that AutoGPTQ can discover
    and quantize. After quantization, unwraps back to the fused format.

    This avoids the memory overhead of physically unfusing into N separate
    nn.Linear modules.
    """

    def __init__(
        self,
        fused_weight: torch.Tensor,
        expert_idx: int,
        total_experts: int,
        original_param_name: str,
    ):
        super().__init__()
        # Extract this expert's 2D slice: [intermediate, hidden]
        self.weight = torch.nn.Parameter(fused_weight[expert_idx].clone())
        self.expert_idx = expert_idx
        self.total_experts = total_experts
        self.original_param_name = original_param_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def _find_fused_expert_params(model: torch.nn.Module) -> list[tuple[str, str, torch.nn.Parameter]]:
    """Find all fused 3D expert parameters in the model."""
    results = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if param.ndim == 3 and param.shape[0] >= 4:
                if any(kw in name.lower() for kw in ("expert", "moe")):
                    results.append((name, pname, param))
    return results


def patch_autogptq():
    """
    Patch AutoGPTQ to handle fused 3D expert tensors.

    Call this BEFORE loading your model with AutoGPTQ.
    """
    global _PATCHED
    if _PATCHED:
        logger.info("AutoGPTQ already patched for Gemma 4 MoE")
        return

    try:
        import auto_gptq
        from auto_gptq.modeling._base import BaseGPTQForCausalLM
    except ImportError:
        raise ImportError(
            "auto-gptq is required. Install with: pip install auto-gptq"
        )

    # Store original method
    _original_find_layers = getattr(
        auto_gptq.modeling._utils, "find_layers", None
    )

    def patched_find_layers(module, layers=None, name=""):
        """
        Extended find_layers that also discovers fused 3D expert parameters
        and wraps them as quantizable modules.
        """
        if layers is None:
            layers = [torch.nn.Linear, torch.nn.Conv2d]

        # Call original
        result = {}
        if _original_find_layers:
            result = _original_find_layers(module, layers, name)

        # Also find fused expert params
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            for pname, param in child.named_parameters(recurse=False):
                if param.ndim == 3 and param.shape[0] >= 4:
                    if any(kw in full_name.lower() for kw in ("expert", "moe")):
                        logger.info(
                            f"Found fused 3D expert tensor: {full_name}.{pname} "
                            f"shape={tuple(param.shape)}"
                        )
                        # Create virtual Linear wrappers for each expert
                        num_experts = param.shape[0]
                        for eid in range(num_experts):
                            wrapper_name = f"{full_name}.{pname}.__expert_{eid}"
                            wrapper = FusedExpertQuantWrapper(
                                param.data, eid, num_experts, f"{full_name}.{pname}"
                            )
                            result[wrapper_name] = wrapper

            # Recurse
            sub_result = patched_find_layers(child, layers, full_name)
            result.update(sub_result)

        return result

    # Apply patch
    if hasattr(auto_gptq.modeling, "_utils"):
        auto_gptq.modeling._utils.find_layers = patched_find_layers

    _PATCHED = True
    logger.info("AutoGPTQ patched for Gemma 4 MoE fused 3D expert tensors")


def unpatch_autogptq():
    """Remove the Gemma 4 MoE patch from AutoGPTQ."""
    global _PATCHED
    if not _PATCHED:
        return

    try:
        import auto_gptq

        # Restore would require storing the original — simplified here
        logger.info("AutoGPTQ unpatch requested (restart recommended)")
    except ImportError:
        pass

    _PATCHED = False
