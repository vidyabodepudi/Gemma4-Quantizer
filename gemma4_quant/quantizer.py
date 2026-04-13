"""
Core quantization engine for fused 3D MoE tensors.

This is the heart of gemma4-quantizer. Instead of unfusing expert weights
into individual nn.Linear layers, this operates directly on the 3D tensor
[num_experts, intermediate_dim, hidden_dim] with per-expert-channel
quantization granularity.

Supported methods:
  - absmax (symmetric): fastest, good for INT8
  - group quantization (symmetric): best quality for INT4 with group_size=128
  - asymmetric min-max: for cases where weight distributions are highly skewed

Supported bit widths: 2, 3, 4, 8
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from gemma4_quant.detector import (
    CheckpointAnalysis,
    FusedExpertDetector,
    TensorKind,
)

logger = logging.getLogger(__name__)


class QuantMethod(Enum):
    """Quantization method."""

    ABSMAX = auto()  # Symmetric per-channel/per-expert absmax
    GROUP = auto()  # Symmetric group quantization (recommended for INT4)
    ASYMMETRIC = auto()  # Asymmetric min-max


class ExportFormat(Enum):
    """Output format for quantized model."""

    SAFETENSORS = auto()
    GGUF = auto()


@dataclass
class QuantConfig:
    """Configuration for quantization."""

    # Core settings
    bits: int = 4  # Quantization bit width (2, 3, 4, or 8)
    method: QuantMethod = QuantMethod.GROUP
    group_size: int = 128  # Group size for GROUP method

    # Expert-specific settings
    expert_bits: Optional[int] = None  # Override bits for expert tensors (None = same as bits)
    skip_experts: bool = False  # Skip quantizing expert tensors entirely
    skip_attention: bool = False  # Skip quantizing attention layers
    skip_embedding: bool = True  # Skip quantizing embeddings (usually recommended)

    # Precision settings
    compute_dtype: torch.dtype = torch.float32  # Dtype for quantization math
    store_dtype: torch.dtype = torch.float16  # Dtype for scales/zero-points

    # Advanced
    symmetric: bool = True  # Symmetric quantization (no zero-point)
    clip_ratio: float = 1.0  # Clip outliers at clip_ratio * max (1.0 = no clipping)

    def __post_init__(self):
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"bits must be 2, 3, 4, or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.expert_bits is None:
            self.expert_bits = self.bits
        if self.expert_bits not in (2, 3, 4, 8):
            raise ValueError(
                f"expert_bits must be 2, 3, 4, or 8, got {self.expert_bits}"
            )

    @property
    def qmin(self) -> int:
        if self.symmetric:
            return -(2 ** (self.bits - 1)) + 1
        return 0

    @property
    def qmax(self) -> int:
        if self.symmetric:
            return 2 ** (self.bits - 1) - 1
        return 2**self.bits - 1

    def qmin_for(self, bits: int) -> int:
        if self.symmetric:
            return -(2 ** (bits - 1)) + 1
        return 0

    def qmax_for(self, bits: int) -> int:
        if self.symmetric:
            return 2 ** (bits - 1) - 1
        return 2**bits - 1


@dataclass
class QuantizedTensor:
    """A quantized tensor with its metadata."""

    name: str
    data: torch.Tensor  # Quantized integer data
    scales: torch.Tensor  # Quantization scales
    zeros: Optional[torch.Tensor]  # Zero-points (None for symmetric)
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    bits: int
    group_size: int
    method: QuantMethod
    is_expert: bool

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs the original tensor."""
        orig_bits = 16  # Assume FP16 original
        return orig_bits / self.bits

    def dequantize(self) -> torch.Tensor:
        """Dequantize back to floating point for validation."""
        data_float = self.data.to(self.scales.dtype)

        if self.method == QuantMethod.ABSMAX:
            # Absmax scales are broadcastable (have trailing dim=1)
            if self.zeros is not None:
                data_float = data_float - self.zeros
            return data_float * self.scales

        # Group quantization: scales have fewer elements along the grouped dim
        # Need to reshape data into groups, multiply, reshape back.
        if self.is_expert and data_float.ndim == 3:
            # 3D: [E, I, H] with scales [E, I, num_groups]
            E, I, H = data_float.shape
            num_groups = self.scales.shape[2]
            group_size = H // num_groups
            # Reshape: [E, I, num_groups, group_size]
            data_grouped = data_float.reshape(E, I, num_groups, group_size)
            scales_expanded = self.scales.unsqueeze(3)  # [E, I, num_groups, 1]
            if self.zeros is not None:
                zeros_expanded = self.zeros.unsqueeze(3)
                data_grouped = data_grouped - zeros_expanded
            result = data_grouped * scales_expanded
            return result.reshape(E, I, H)
        elif data_float.ndim == 2:
            # 2D: [O, I] with scales [O, num_groups]
            O, I = data_float.shape
            num_groups = self.scales.shape[1]
            group_size = I // num_groups
            # Reshape: [O, num_groups, group_size]
            data_grouped = data_float.reshape(O, num_groups, group_size)
            scales_expanded = self.scales.unsqueeze(2)  # [O, num_groups, 1]
            if self.zeros is not None:
                zeros_expanded = self.zeros.unsqueeze(2)
                data_grouped = data_grouped - zeros_expanded
            result = data_grouped * scales_expanded
            return result.reshape(O, I)
        else:
            # Fallback: try direct multiply
            if self.zeros is not None:
                data_float = data_float - self.zeros
            return data_float * self.scales


@dataclass
class QuantizationResult:
    """Result of quantizing an entire model."""

    quantized_tensors: dict[str, QuantizedTensor] = field(default_factory=dict)
    passthrough_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    config: QuantConfig = field(default_factory=QuantConfig)
    analysis: Optional[CheckpointAnalysis] = None
    errors: list[str] = field(default_factory=list)

    @property
    def total_original_bytes(self) -> int:
        total = 0
        for qt in self.quantized_tensors.values():
            params = 1
            for dim in qt.original_shape:
                params *= dim
            total += params * 2  # Assume FP16
        for t in self.passthrough_tensors.values():
            total += t.nelement() * t.element_size()
        return total

    @property
    def total_quantized_bytes(self) -> int:
        total = 0
        for qt in self.quantized_tensors.values():
            # Quantized data
            total += qt.data.nelement()  # INT8 packing
            # Scales
            total += qt.scales.nelement() * qt.scales.element_size()
            # Zeros
            if qt.zeros is not None:
                total += qt.zeros.nelement() * qt.zeros.element_size()
        for t in self.passthrough_tensors.values():
            total += t.nelement() * t.element_size()
        return total

    @property
    def compression_ratio(self) -> float:
        if self.total_quantized_bytes == 0:
            return 0.0
        return self.total_original_bytes / self.total_quantized_bytes


class Gemma4Quantizer:
    """
    Native 3D tensor quantizer for Gemma 4 MoE models.

    This quantizer handles fused expert tensors [num_experts, intermediate, hidden]
    directly without unfusing them, using per-expert-channel quantization.

    Usage:
        config = QuantConfig(bits=4, method=QuantMethod.GROUP, group_size=128)
        quantizer = Gemma4Quantizer(config)

        # From safetensors checkpoint:
        result = quantizer.quantize_checkpoint("/path/to/model/")

        # From state_dict:
        result = quantizer.quantize_state_dict(model.state_dict())

        # Export:
        quantizer.export_safetensors(result, "/path/to/output/")
    """

    def __init__(
        self,
        config: QuantConfig,
        detector: Optional[FusedExpertDetector] = None,
    ):
        self.config = config
        self.detector = detector or FusedExpertDetector()

    # ------------------------------------------------------------------
    # Core quantization methods operating directly on 3D tensors
    # ------------------------------------------------------------------

    def quantize_3d_expert_absmax(
        self,
        weight: torch.Tensor,
        bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric absmax quantization of a fused 3D expert tensor.

        Operates per-expert, per-output-channel:
          weight: [num_experts, intermediate_dim, hidden_dim]
          scales: [num_experts, intermediate_dim, 1]

        Each expert gets independent scales, preserving the fused layout.
        """
        assert weight.ndim == 3, f"Expected 3D tensor, got {weight.ndim}D"
        E, I, H = weight.shape

        qmin = self.config.qmin_for(bits)
        qmax = self.config.qmax_for(bits)

        w = weight.to(self.config.compute_dtype)

        # Per-expert, per-row absmax: [E, I, 1]
        amax = w.abs().amax(dim=2, keepdim=True)
        amax = amax.clamp(min=1e-10)  # avoid division by zero

        if self.config.clip_ratio < 1.0:
            amax = amax * self.config.clip_ratio

        scales = amax / qmax  # [E, I, 1]

        # Quantize
        w_q = (w / scales).round().clamp(qmin, qmax).to(torch.int8)

        return w_q, scales.to(self.config.store_dtype)

    def quantize_3d_expert_group(
        self,
        weight: torch.Tensor,
        bits: int,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric group quantization of a fused 3D expert tensor.

        Groups along the hidden_dim (last axis) for finer granularity:
          weight: [num_experts, intermediate_dim, hidden_dim]
          → reshape to [E, I, H // group_size, group_size]
          scales: [E, I, H // group_size]

        This gives each expert its own per-row, per-group scales.
        """
        assert weight.ndim == 3, f"Expected 3D tensor, got {weight.ndim}D"
        E, I, H = weight.shape

        if H % group_size != 0:
            # Pad hidden dim to be divisible by group_size
            pad_size = group_size - (H % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))
            H = H + pad_size
            logger.debug(
                f"Padded hidden dim from {H - pad_size} to {H} "
                f"(group_size={group_size})"
            )

        qmin = self.config.qmin_for(bits)
        qmax = self.config.qmax_for(bits)

        w = weight.to(self.config.compute_dtype)

        # Reshape: [E, I, H] → [E, I, num_groups, group_size]
        num_groups = H // group_size
        w = w.reshape(E, I, num_groups, group_size)

        # Per-expert, per-row, per-group absmax: [E, I, num_groups, 1]
        amax = w.abs().amax(dim=3, keepdim=True)
        amax = amax.clamp(min=1e-10)

        if self.config.clip_ratio < 1.0:
            amax = amax * self.config.clip_ratio

        scales = amax / qmax  # [E, I, num_groups, 1]

        # Quantize
        w_q = (w / scales).round().clamp(qmin, qmax).to(torch.int8)

        # Reshape back: [E, I, H]
        w_q = w_q.reshape(E, I, H)
        scales = scales.squeeze(3)  # [E, I, num_groups]

        return w_q, scales.to(self.config.store_dtype)

    def quantize_3d_expert_asymmetric(
        self,
        weight: torch.Tensor,
        bits: int,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Asymmetric min-max quantization of a fused 3D expert tensor.

        Returns (quantized_data, scales, zero_points).
        """
        assert weight.ndim == 3, f"Expected 3D tensor, got {weight.ndim}D"
        E, I, H = weight.shape

        if H % group_size != 0:
            pad_size = group_size - (H % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))
            H = H + pad_size

        qmin = self.config.qmin_for(bits)
        qmax = self.config.qmax_for(bits)

        w = weight.to(self.config.compute_dtype)

        num_groups = H // group_size
        w = w.reshape(E, I, num_groups, group_size)

        w_min = w.amin(dim=3, keepdim=True)
        w_max = w.amax(dim=3, keepdim=True)

        scales = (w_max - w_min) / (qmax - qmin)
        scales = scales.clamp(min=1e-10)
        zeros = qmin - (w_min / scales)

        w_q = (w / scales + zeros).round().clamp(qmin, qmax).to(torch.int8)

        w_q = w_q.reshape(E, I, H)
        scales = scales.squeeze(3)
        zeros = zeros.squeeze(3)

        return (
            w_q,
            scales.to(self.config.store_dtype),
            zeros.to(self.config.store_dtype),
        )

    # ------------------------------------------------------------------
    # Standard 2D quantization (for attention, etc.)
    # ------------------------------------------------------------------

    def quantize_2d_linear(
        self,
        weight: torch.Tensor,
        bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard per-channel symmetric quantization for 2D weight matrices."""
        assert weight.ndim == 2, f"Expected 2D tensor, got {weight.ndim}D"

        if self.config.method == QuantMethod.ABSMAX:
            return self._quantize_2d_absmax(weight, bits)
        elif self.config.method == QuantMethod.GROUP:
            return self._quantize_2d_group(weight, bits, self.config.group_size)
        else:
            q, s, _ = self._quantize_2d_asym(weight, bits, self.config.group_size)
            return q, s

    def _quantize_2d_absmax(
        self, weight: torch.Tensor, bits: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        O, I = weight.shape
        qmin = self.config.qmin_for(bits)
        qmax = self.config.qmax_for(bits)

        w = weight.to(self.config.compute_dtype)
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        if self.config.clip_ratio < 1.0:
            amax *= self.config.clip_ratio
        scales = amax / qmax
        w_q = (w / scales).round().clamp(qmin, qmax).to(torch.int8)
        return w_q, scales.to(self.config.store_dtype)

    def _quantize_2d_group(
        self, weight: torch.Tensor, bits: int, group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        O, I = weight.shape
        if I % group_size != 0:
            pad = group_size - (I % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad))
            I = I + pad

        qmin = self.config.qmin_for(bits)
        qmax = self.config.qmax_for(bits)

        w = weight.to(self.config.compute_dtype)
        num_groups = I // group_size
        w = w.reshape(O, num_groups, group_size)
        amax = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-10)
        if self.config.clip_ratio < 1.0:
            amax *= self.config.clip_ratio
        scales = amax / qmax
        w_q = (w / scales).round().clamp(qmin, qmax).to(torch.int8)
        w_q = w_q.reshape(O, I)
        scales = scales.squeeze(2)
        return w_q, scales.to(self.config.store_dtype)

    def _quantize_2d_asym(
        self, weight: torch.Tensor, bits: int, group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        O, I = weight.shape
        if I % group_size != 0:
            pad = group_size - (I % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad))
            I = I + pad

        qmin = self.config.qmin_for(bits)
        qmax = self.config.qmax_for(bits)

        w = weight.to(self.config.compute_dtype)
        num_groups = I // group_size
        w = w.reshape(O, num_groups, group_size)
        w_min = w.amin(dim=2, keepdim=True)
        w_max = w.amax(dim=2, keepdim=True)
        scales = ((w_max - w_min) / (qmax - qmin)).clamp(min=1e-10)
        zeros = qmin - (w_min / scales)
        w_q = (w / scales + zeros).round().clamp(qmin, qmax).to(torch.int8)
        w_q = w_q.reshape(O, I)
        scales = scales.squeeze(2)
        zeros = zeros.squeeze(2)
        return w_q, scales.to(self.config.store_dtype), zeros.to(self.config.store_dtype)

    # ------------------------------------------------------------------
    # High-level quantization pipeline
    # ------------------------------------------------------------------

    def quantize_tensor(
        self, name: str, tensor: torch.Tensor, kind: TensorKind
    ) -> Optional[QuantizedTensor]:
        """Quantize a single tensor based on its classification."""

        # Skip norms and biases — always keep full precision
        if kind in (TensorKind.NORM_1D, TensorKind.OTHER):
            return None

        # Skip embeddings if configured
        if kind == TensorKind.EMBEDDING and self.config.skip_embedding:
            return None

        # Skip attention if configured
        if kind == TensorKind.LINEAR_2D and self.config.skip_attention:
            # Heuristic: if "attention" / "self_attn" in name, skip
            if any(kw in name.lower() for kw in ("attention", "self_attn", "attn")):
                return None

        # Skip experts if configured
        if kind == TensorKind.FUSED_EXPERT_3D and self.config.skip_experts:
            return None

        is_expert = kind == TensorKind.FUSED_EXPERT_3D
        bits = self.config.expert_bits if is_expert else self.config.bits

        zeros = None

        if kind == TensorKind.FUSED_EXPERT_3D:
            if self.config.method == QuantMethod.ABSMAX:
                data, scales = self.quantize_3d_expert_absmax(tensor, bits)
            elif self.config.method == QuantMethod.GROUP:
                data, scales = self.quantize_3d_expert_group(
                    tensor, bits, self.config.group_size
                )
            elif self.config.method == QuantMethod.ASYMMETRIC:
                data, scales, zeros = self.quantize_3d_expert_asymmetric(
                    tensor, bits, self.config.group_size
                )
            else:
                raise ValueError(f"Unknown method: {self.config.method}")

        elif kind in (TensorKind.LINEAR_2D, TensorKind.EMBEDDING):
            data, scales = self.quantize_2d_linear(tensor, bits)

        else:
            return None

        return QuantizedTensor(
            name=name,
            data=data,
            scales=scales,
            zeros=zeros,
            original_shape=tuple(tensor.shape),
            original_dtype=tensor.dtype,
            bits=bits,
            group_size=self.config.group_size,
            method=self.config.method,
            is_expert=is_expert,
        )

    def quantize_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        show_progress: bool = True,
    ) -> QuantizationResult:
        """Quantize an entire model's state_dict."""

        # First, analyze
        analysis = self.detector.analyze_state_dict(state_dict)
        logger.info(analysis.summary())

        result = QuantizationResult(config=self.config, analysis=analysis)

        items = list(state_dict.items())
        if show_progress:
            items = tqdm(items, desc="Quantizing", unit="tensor")

        for name, tensor in items:
            kind = self.detector.classify_tensor(name, tuple(tensor.shape))

            try:
                qt = self.quantize_tensor(name, tensor, kind)
                if qt is not None:
                    result.quantized_tensors[name] = qt
                else:
                    # Keep as passthrough (norms, biases, skipped tensors)
                    result.passthrough_tensors[name] = tensor
            except Exception as e:
                msg = f"Error quantizing {name}: {e}"
                logger.error(msg)
                result.errors.append(msg)
                result.passthrough_tensors[name] = tensor

        return result

    def quantize_checkpoint(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        show_progress: bool = True,
    ) -> QuantizationResult:
        """
        Quantize a safetensors checkpoint (single file or sharded directory).
        Processes tensors one shard at a time to minimize memory usage.
        """
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError(
                "safetensors is required. Install with: pip install safetensors"
            )

        path = Path(checkpoint_path)
        if path.is_dir():
            shard_files = sorted(path.glob("*.safetensors"))
        else:
            shard_files = [path]

        # First pass: analyze metadata
        analysis = self.detector.analyze_safetensors(path, device=device)
        logger.info(analysis.summary())

        result = QuantizationResult(config=self.config, analysis=analysis)

        for shard_path in shard_files:
            logger.info(f"Processing shard: {shard_path.name}")

            with safe_open(str(shard_path), framework="pt", device=device) as f:
                tensor_names = list(f.keys())
                if show_progress:
                    tensor_names = tqdm(
                        tensor_names,
                        desc=f"  {shard_path.name}",
                        unit="tensor",
                    )

                for name in tensor_names:
                    tensor = f.get_tensor(name)
                    kind = self.detector.classify_tensor(name, tuple(tensor.shape))

                    try:
                        qt = self.quantize_tensor(name, tensor, kind)
                        if qt is not None:
                            result.quantized_tensors[name] = qt
                        else:
                            result.passthrough_tensors[name] = tensor
                    except Exception as e:
                        msg = f"Error quantizing {name}: {e}"
                        logger.error(msg)
                        result.errors.append(msg)
                        result.passthrough_tensors[name] = tensor

        return result
