"""
Calibration engine for activation-aware quantization.

Collects per-expert activation statistics during a forward pass to guide
quantization decisions:
  - Router-guided sampling: weight calibration effort by expert utilization
  - Smooth quantization: migrate quantization difficulty from activations to weights
  - Optimal clipping: find the clip ratio that minimizes quantization error
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ExpertStats:
    """Activation statistics for a single expert."""

    expert_id: int
    activation_count: int = 0  # How many tokens routed to this expert
    weight_absmax: float = 0.0
    activation_absmax: float = 0.0
    weight_mean: float = 0.0
    weight_std: float = 0.0
    activation_mean: float = 0.0
    activation_std: float = 0.0


@dataclass
class LayerCalibrationData:
    """Calibration data for one MoE layer."""

    layer_idx: int
    expert_stats: dict[int, ExpertStats] = field(default_factory=dict)
    router_probs: Optional[torch.Tensor] = None  # [num_calibration_tokens, num_experts]
    optimal_clip_ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """Complete calibration data for the model."""

    layers: dict[int, LayerCalibrationData] = field(default_factory=dict)
    num_calibration_samples: int = 0
    num_tokens_processed: int = 0

    def get_expert_utilization(self, layer_idx: int) -> dict[int, float]:
        """
        Get expert utilization ratios for a layer.
        Returns dict mapping expert_id → fraction of tokens routed to it.
        """
        if layer_idx not in self.layers:
            return {}

        layer = self.layers[layer_idx]
        total = sum(s.activation_count for s in layer.expert_stats.values())
        if total == 0:
            return {}

        return {
            eid: s.activation_count / total
            for eid, s in layer.expert_stats.items()
        }

    def suggest_expert_bits(
        self,
        layer_idx: int,
        default_bits: int = 4,
        high_util_threshold: float = 0.05,
        high_util_bits: int = 4,
        low_util_bits: int = 3,
    ) -> dict[int, int]:
        """
        Suggest per-expert bit widths based on utilization.
        Frequently-used experts get higher precision.
        """
        utilization = self.get_expert_utilization(layer_idx)
        if not utilization:
            return {}

        return {
            eid: high_util_bits if util >= high_util_threshold else low_util_bits
            for eid, util in utilization.items()
        }


class CalibrationEngine:
    """
    Collects calibration data by running forward passes on sample data.

    Usage:
        engine = CalibrationEngine(model, tokenizer)
        result = engine.calibrate(
            calibration_texts=["sample text 1", "sample text 2", ...],
            num_samples=128,
        )

        # Use calibration to set optimal clip ratios
        for layer_idx, layer_data in result.layers.items():
            for tensor_name, clip_ratio in layer_data.optimal_clip_ratios.items():
                print(f"  {tensor_name}: clip_ratio={clip_ratio:.4f}")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer=None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._hooks = []
        self._layer_data: dict[int, LayerCalibrationData] = {}

    def _register_hooks(self):
        """Register forward hooks on MoE layers to capture routing decisions."""
        for name, module in self.model.named_modules():
            # Look for the router/gate module in MoE layers
            if any(
                kw in name.lower()
                for kw in ("block_sparse_moe", "sparse_moe", "moe_gate")
            ):
                # Extract layer index
                import re

                layer_match = re.search(r"layers\.(\d+)", name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))

                    if layer_idx not in self._layer_data:
                        self._layer_data[layer_idx] = LayerCalibrationData(
                            layer_idx=layer_idx
                        )

                    hook = module.register_forward_hook(
                        self._make_hook(layer_idx, name)
                    )
                    self._hooks.append(hook)

    def _make_hook(self, layer_idx: int, module_name: str):
        """Create a forward hook that captures expert routing statistics."""
        layer_data = self._layer_data[layer_idx]

        def hook_fn(module, input, output):
            # Try to extract router logits/probs
            # Gemma 4 MoE router typically outputs (hidden_states, router_logits)
            if isinstance(output, tuple) and len(output) >= 2:
                router_logits = output[1]
                if isinstance(router_logits, torch.Tensor) and router_logits.ndim >= 2:
                    # router_logits: [batch*seq, num_experts]
                    probs = torch.softmax(router_logits.float(), dim=-1)
                    top_experts = probs.argmax(dim=-1)

                    for expert_id in range(probs.shape[-1]):
                        if expert_id not in layer_data.expert_stats:
                            layer_data.expert_stats[expert_id] = ExpertStats(
                                expert_id=expert_id
                            )

                        count = (top_experts == expert_id).sum().item()
                        layer_data.expert_stats[expert_id].activation_count += int(
                            count
                        )

        return hook_fn

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def calibrate(
        self,
        calibration_texts: Optional[list[str]] = None,
        calibration_dataset: Optional[str] = "wikitext",
        num_samples: int = 128,
        max_length: int = 2048,
    ) -> CalibrationResult:
        """
        Run calibration forward passes to collect expert statistics.

        Args:
            calibration_texts: List of text strings. If None, uses calibration_dataset.
            calibration_dataset: HuggingFace dataset name (fallback).
            num_samples: Number of samples to process.
            max_length: Maximum sequence length.

        Returns:
            CalibrationResult with per-layer, per-expert statistics.
        """
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer required for calibration. "
                "Pass tokenizer to CalibrationEngine constructor."
            )

        # Get calibration texts
        if calibration_texts is None:
            calibration_texts = self._load_calibration_dataset(
                calibration_dataset, num_samples
            )

        self.model.eval()
        self._register_hooks()

        result = CalibrationResult()
        total_tokens = 0

        try:
            for i, text in enumerate(calibration_texts[:num_samples]):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                ).to(self.device)

                self.model(**inputs)
                total_tokens += inputs["input_ids"].shape[1]

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Calibration progress: {i + 1}/{min(num_samples, len(calibration_texts))} "
                        f"samples, {total_tokens} tokens"
                    )

        finally:
            self._remove_hooks()

        result.layers = self._layer_data
        result.num_calibration_samples = min(num_samples, len(calibration_texts))
        result.num_tokens_processed = total_tokens

        # Compute optimal clip ratios per expert tensor
        self._compute_optimal_clips(result)

        return result

    def _load_calibration_dataset(
        self, dataset_name: str, num_samples: int
    ) -> list[str]:
        """Load calibration texts from a HuggingFace dataset."""
        try:
            from datasets import load_dataset

            if dataset_name == "wikitext":
                ds = load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split="test"
                )
                texts = [
                    t["text"]
                    for t in ds
                    if t["text"].strip() and len(t["text"]) > 100
                ]
            else:
                ds = load_dataset(dataset_name, split="train")
                text_col = "text" if "text" in ds.column_names else ds.column_names[0]
                texts = [t[text_col] for t in ds if len(t[text_col]) > 100]

            return texts[:num_samples]
        except ImportError:
            raise ImportError(
                "datasets library required for calibration. "
                "Install with: pip install datasets"
            )

    def _compute_optimal_clips(self, result: CalibrationResult):
        """
        Compute optimal clip ratios using grid search over quantization error.
        Only runs if we have the actual weight tensors accessible.
        """
        # This is a lightweight version — full implementation would search
        # over clip_ratio ∈ [0.7, 1.0] for each expert tensor and pick
        # the value that minimizes MSE between quantized and original.
        for layer_idx, layer_data in result.layers.items():
            for name, module in self.model.named_modules():
                if f"layers.{layer_idx}" in name and "experts" in name:
                    for pname, param in module.named_parameters():
                        if param.ndim == 3:
                            clip = self._grid_search_clip(param.data)
                            full_name = f"{name}.{pname}"
                            layer_data.optimal_clip_ratios[full_name] = clip

    @staticmethod
    def _grid_search_clip(
        weight: torch.Tensor,
        candidates: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00),
        bits: int = 4,
    ) -> float:
        """Find the clip ratio that minimizes quantization MSE."""
        qmax = 2 ** (bits - 1) - 1
        best_ratio = 1.0
        best_mse = float("inf")

        w = weight.float()

        for ratio in candidates:
            amax = w.abs().amax(dim=-1, keepdim=True) * ratio
            amax = amax.clamp(min=1e-10)
            scale = amax / qmax
            w_q = (w / scale).round().clamp(-qmax, qmax)
            w_deq = w_q * scale
            mse = (w - w_deq).pow(2).mean().item()

            if mse < best_mse:
                best_mse = mse
                best_ratio = ratio

        return best_ratio
