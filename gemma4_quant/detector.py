"""
Fused 3D Expert Tensor Detector.

Scans a Gemma 4 MoE checkpoint (safetensors or state_dict) to identify:
  1. Fused 3D expert tensors — shape [num_experts, dim1, dim2]
  2. Standard 2D weight tensors — shape [out, in]
  3. 1D bias / norm tensors

This information drives the quantizer's dispatch logic: 3D tensors get
per-expert-channel quantization, 2D tensors get standard quantization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as st_load_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class TensorKind(Enum):
    """Classification of tensors in a Gemma 4 MoE checkpoint."""

    FUSED_EXPERT_3D = auto()  # [num_experts, intermediate, hidden]
    LINEAR_2D = auto()  # Standard nn.Linear weight
    EMBEDDING = auto()  # Token embeddings
    NORM_1D = auto()  # RMSNorm / LayerNorm scales
    OTHER = auto()  # Biases, scalars, etc.


@dataclass
class TensorInfo:
    """Metadata about a single tensor in the checkpoint."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    kind: TensorKind
    num_experts: Optional[int] = None  # Only for FUSED_EXPERT_3D
    size_bytes: int = 0

    @property
    def num_parameters(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass
class CheckpointAnalysis:
    """Complete analysis of a Gemma 4 MoE checkpoint."""

    tensors: list[TensorInfo] = field(default_factory=list)
    total_params: int = 0
    expert_params: int = 0
    non_expert_params: int = 0
    num_experts_detected: int = 0
    num_moe_layers: int = 0
    model_config: dict = field(default_factory=dict)

    @property
    def expert_param_ratio(self) -> float:
        if self.total_params == 0:
            return 0.0
        return self.expert_params / self.total_params

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Gemma 4 MoE Checkpoint Analysis",
            "=" * 60,
            f"  Total parameters:    {self.total_params:>14,}",
            f"  Expert parameters:   {self.expert_params:>14,} ({self.expert_param_ratio:.1%})",
            f"  Non-expert params:   {self.non_expert_params:>14,} ({1 - self.expert_param_ratio:.1%})",
            f"  Detected experts:    {self.num_experts_detected:>14}",
            f"  MoE layers:          {self.num_moe_layers:>14}",
            "",
            "  Tensor breakdown:",
        ]

        kind_counts: dict[TensorKind, int] = {}
        kind_params: dict[TensorKind, int] = {}
        for t in self.tensors:
            kind_counts[t.kind] = kind_counts.get(t.kind, 0) + 1
            kind_params[t.kind] = kind_params.get(t.kind, 0) + t.num_parameters

        for kind in TensorKind:
            count = kind_counts.get(kind, 0)
            params = kind_params.get(kind, 0)
            if count > 0:
                lines.append(f"    {kind.name:<20} {count:>4} tensors, {params:>14,} params")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patterns for identifying fused expert tensors in Gemma 4
# ---------------------------------------------------------------------------

# Gemma 4 MoE stores experts in layers like:
#   model.layers.{N}.block_sparse_moe.experts.gate_up_proj  -> [num_experts, 2*intermediate, hidden]
#   model.layers.{N}.block_sparse_moe.experts.down_proj     -> [num_experts, hidden, intermediate]
# The key indicator: 3D tensor under a path containing "experts" or "moe"
FUSED_EXPERT_PATTERNS = [
    re.compile(r".*block_sparse_moe\.experts\.\w+"),
    re.compile(r".*moe\.experts\.\w+"),
    re.compile(r".*sparse_moe\.experts\.\w+"),
    # Generic fallback: any 3D tensor with "expert" in the name
    re.compile(r".*expert.*"),
]

EMBEDDING_PATTERNS = [
    re.compile(r".*embed_tokens.*"),
    re.compile(r".*wte.*"),
    re.compile(r".*token_embedding.*"),
]

NORM_PATTERNS = [
    re.compile(r".*norm.*weight"),
    re.compile(r".*layernorm.*"),
    re.compile(r".*rmsnorm.*"),
]


class FusedExpertDetector:
    """
    Detects and classifies tensors in a Gemma 4 MoE checkpoint.

    Usage:
        detector = FusedExpertDetector()

        # From a safetensors file:
        analysis = detector.analyze_safetensors("/path/to/model.safetensors")

        # From a state_dict:
        analysis = detector.analyze_state_dict(model.state_dict())

        print(analysis.summary())
    """

    def __init__(
        self,
        expert_patterns: Optional[list[re.Pattern]] = None,
        min_expert_dim: int = 4,
    ):
        """
        Args:
            expert_patterns: Custom regex patterns for identifying fused expert tensors.
                             Defaults to Gemma 4 patterns.
            min_expert_dim: Minimum value for dim-0 to be considered "multiple experts"
                            (avoids false positives on small 3D tensors).
        """
        self.expert_patterns = expert_patterns or FUSED_EXPERT_PATTERNS
        self.min_expert_dim = min_expert_dim

    def classify_tensor(self, name: str, shape: tuple[int, ...]) -> TensorKind:
        """Classify a single tensor by name and shape."""
        ndim = len(shape)

        # 3D tensors matching expert patterns → fused expert
        if ndim == 3 and shape[0] >= self.min_expert_dim:
            for pattern in self.expert_patterns:
                if pattern.match(name):
                    return TensorKind.FUSED_EXPERT_3D

        # Embeddings
        if ndim == 2:
            for pattern in EMBEDDING_PATTERNS:
                if pattern.match(name):
                    return TensorKind.EMBEDDING

        # Norm parameters
        if ndim == 1:
            for pattern in NORM_PATTERNS:
                if pattern.match(name):
                    return TensorKind.NORM_1D

        # Standard linear weights
        if ndim == 2:
            return TensorKind.LINEAR_2D

        # 1D that didn't match norm
        if ndim == 1:
            return TensorKind.NORM_1D

        return TensorKind.OTHER

    def analyze_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> CheckpointAnalysis:
        """Analyze a PyTorch state_dict."""
        analysis = CheckpointAnalysis()
        moe_layer_indices: set[int] = set()
        expert_counts: set[int] = set()

        for name, tensor in state_dict.items():
            shape = tuple(tensor.shape)
            kind = self.classify_tensor(name, shape)

            info = TensorInfo(
                name=name,
                shape=shape,
                dtype=tensor.dtype,
                kind=kind,
                num_experts=shape[0] if kind == TensorKind.FUSED_EXPERT_3D else None,
                size_bytes=tensor.nelement() * tensor.element_size(),
            )
            analysis.tensors.append(info)

            params = info.num_parameters
            analysis.total_params += params

            if kind == TensorKind.FUSED_EXPERT_3D:
                analysis.expert_params += params
                expert_counts.add(shape[0])
                # Extract layer index
                layer_match = re.search(r"layers\.(\d+)", name)
                if layer_match:
                    moe_layer_indices.add(int(layer_match.group(1)))
            else:
                analysis.non_expert_params += params

        analysis.num_moe_layers = len(moe_layer_indices)
        if expert_counts:
            analysis.num_experts_detected = max(expert_counts)

        return analysis

    def analyze_safetensors(
        self, path: str | Path, device: str = "cpu"
    ) -> CheckpointAnalysis:
        """
        Analyze a safetensors file without loading all weights into memory.
        Only reads tensor metadata (names, shapes, dtypes).
        """
        if not HAS_SAFETENSORS:
            raise ImportError(
                "safetensors is required for analyze_safetensors(). "
                "Install with: pip install safetensors"
            )

        path = Path(path)
        analysis = CheckpointAnalysis()
        moe_layer_indices: set[int] = set()
        expert_counts: set[int] = set()

        # Handle single file or sharded checkpoint directory
        if path.is_dir():
            shard_files = sorted(path.glob("*.safetensors"))
        else:
            shard_files = [path]

        for shard_path in shard_files:
            with safe_open(str(shard_path), framework="pt", device=device) as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    shape = tuple(tensor.shape)
                    kind = self.classify_tensor(name, shape)

                    info = TensorInfo(
                        name=name,
                        shape=shape,
                        dtype=tensor.dtype,
                        kind=kind,
                        num_experts=shape[0]
                        if kind == TensorKind.FUSED_EXPERT_3D
                        else None,
                        size_bytes=tensor.nelement() * tensor.element_size(),
                    )
                    analysis.tensors.append(info)

                    params = info.num_parameters
                    analysis.total_params += params

                    if kind == TensorKind.FUSED_EXPERT_3D:
                        analysis.expert_params += params
                        expert_counts.add(shape[0])
                        layer_match = re.search(r"layers\.(\d+)", name)
                        if layer_match:
                            moe_layer_indices.add(int(layer_match.group(1)))
                    else:
                        analysis.non_expert_params += params

        analysis.num_moe_layers = len(moe_layer_indices)
        if expert_counts:
            analysis.num_experts_detected = max(expert_counts)

        return analysis

    def analyze_safetensors_metadata_only(
        self, path: str | Path
    ) -> CheckpointAnalysis:
        """
        Ultra-lightweight analysis using only safetensors metadata.
        Does NOT load any tensor data — just reads the header.
        """
        if not HAS_SAFETENSORS:
            raise ImportError(
                "safetensors is required. Install with: pip install safetensors"
            )

        import json
        import struct

        path = Path(path)
        if path.is_dir():
            shard_files = sorted(path.glob("*.safetensors"))
        else:
            shard_files = [path]

        analysis = CheckpointAnalysis()
        moe_layer_indices: set[int] = set()
        expert_counts: set[int] = set()

        DTYPE_SIZES = {
            "F32": 4,
            "F16": 2,
            "BF16": 2,
            "I8": 1,
            "I16": 2,
            "I32": 4,
            "I64": 8,
            "U8": 1,
            "F64": 8,
        }

        DTYPE_MAP = {
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I8": torch.int8,
            "I32": torch.int32,
            "I64": torch.int64,
        }

        for shard_path in shard_files:
            with open(shard_path, "rb") as f:
                # Read header size (first 8 bytes, little-endian uint64)
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes)

            for name, meta in header.items():
                if name == "__metadata__":
                    continue

                shape = tuple(meta["shape"])
                dtype_str = meta["dtype"]
                dtype = DTYPE_MAP.get(dtype_str, torch.float32)
                kind = self.classify_tensor(name, shape)

                num_params = 1
                for dim in shape:
                    num_params *= dim

                elem_size = DTYPE_SIZES.get(dtype_str, 2)

                info = TensorInfo(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    kind=kind,
                    num_experts=shape[0]
                    if kind == TensorKind.FUSED_EXPERT_3D
                    else None,
                    size_bytes=num_params * elem_size,
                )
                analysis.tensors.append(info)
                analysis.total_params += num_params

                if kind == TensorKind.FUSED_EXPERT_3D:
                    analysis.expert_params += num_params
                    expert_counts.add(shape[0])
                    layer_match = re.search(r"layers\.(\d+)", name)
                    if layer_match:
                        moe_layer_indices.add(int(layer_match.group(1)))
                else:
                    analysis.non_expert_params += num_params

        analysis.num_moe_layers = len(moe_layer_indices)
        if expert_counts:
            analysis.num_experts_detected = max(expert_counts)

        return analysis
