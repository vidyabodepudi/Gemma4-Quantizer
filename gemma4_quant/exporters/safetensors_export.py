"""
SafeTensors exporter for quantized Gemma 4 MoE models.

Exports quantized weights in a format compatible with vLLM's Marlin backend
and other inference engines that support quantized safetensors.

Naming conventions for quantized tensors:
  - weight: {original_name}.qweight  (packed INT data)
  - scales: {original_name}.scales
  - zeros:  {original_name}.qzeros   (only for asymmetric)
  - meta:   {original_name}.meta     (JSON metadata as string tensor)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from gemma4_quant.quantizer import QuantizationResult, QuantizedTensor

logger = logging.getLogger(__name__)


def _pack_int4_to_int32(data: torch.Tensor) -> torch.Tensor:
    """
    Pack INT4 values into INT32 for efficient storage.
    8 INT4 values packed per INT32.
    """
    data = data.to(torch.int8)
    # Ensure the last dim is divisible by 8
    *prefix, last = data.shape
    if last % 8 != 0:
        pad = 8 - (last % 8)
        data = torch.nn.functional.pad(data, (0, pad))
        last = last + pad

    data = data.reshape(*prefix, last // 8, 8)
    # Shift each nibble into position within int32
    shifts = torch.arange(8, device=data.device) * 4
    packed = (data.to(torch.int32) & 0xF) << shifts
    packed = packed.sum(dim=-1)
    return packed


def export_safetensors(
    result: QuantizationResult,
    output_dir: str | Path,
    max_shard_size_gb: float = 4.0,
    pack_int4: bool = True,
    include_metadata: bool = True,
) -> list[Path]:
    """
    Export quantized model to safetensors format.

    Args:
        result: Quantization result from Gemma4Quantizer.
        output_dir: Directory to write safetensors files.
        max_shard_size_gb: Maximum shard size in GB.
        pack_int4: Whether to pack INT4 into INT32 (saves 50% vs INT8 storage).
        include_metadata: Include quantization metadata in the safetensors header.

    Returns:
        List of paths to written safetensors files.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "safetensors required for export. Install with: pip install safetensors"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the tensor dict
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    # Global metadata
    if include_metadata:
        metadata["quantization_config"] = json.dumps(
            {
                "quant_method": result.config.method.name.lower(),
                "bits": result.config.bits,
                "expert_bits": result.config.expert_bits,
                "group_size": result.config.group_size,
                "symmetric": result.config.symmetric,
                "format": "gemma4-quantizer",
                "version": "0.1.0",
            }
        )

    # Add quantized tensors
    for name, qt in result.quantized_tensors.items():
        if pack_int4 and qt.bits == 4:
            tensors[f"{name}.qweight"] = _pack_int4_to_int32(qt.data)
        else:
            tensors[f"{name}.qweight"] = qt.data

        tensors[f"{name}.scales"] = qt.scales

        if qt.zeros is not None:
            tensors[f"{name}.qzeros"] = qt.zeros

        # Per-tensor metadata
        if include_metadata:
            metadata[f"{name}.quant_meta"] = json.dumps(
                {
                    "original_shape": list(qt.original_shape),
                    "original_dtype": str(qt.original_dtype),
                    "bits": qt.bits,
                    "group_size": qt.group_size,
                    "is_expert": qt.is_expert,
                    "packed_int4": pack_int4 and qt.bits == 4,
                }
            )

    # Add passthrough tensors (norms, biases, etc.)
    for name, tensor in result.passthrough_tensors.items():
        tensors[name] = tensor

    # Shard if needed
    max_shard_bytes = int(max_shard_size_gb * 1024**3)
    shards = _create_shards(tensors, max_shard_bytes)

    written_files = []
    if len(shards) == 1:
        filepath = output_dir / "model.safetensors"
        save_file(shards[0], str(filepath), metadata=metadata)
        written_files.append(filepath)
        logger.info(f"Saved single shard: {filepath}")
    else:
        index = {"metadata": metadata, "weight_map": {}}
        for i, shard in enumerate(shards):
            filename = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
            filepath = output_dir / filename
            shard_metadata = metadata if i == 0 else {}
            save_file(shard, str(filepath), metadata=shard_metadata)
            written_files.append(filepath)
            for tensor_name in shard:
                index["weight_map"][tensor_name] = filename
            logger.info(f"Saved shard {i + 1}/{len(shards)}: {filepath}")

        # Write index
        index_path = output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        written_files.append(index_path)

    # Write quantization config
    config_path = output_dir / "quantize_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "quant_method": result.config.method.name.lower(),
                "bits": result.config.bits,
                "expert_bits": result.config.expert_bits,
                "group_size": result.config.group_size,
                "symmetric": result.config.symmetric,
                "clip_ratio": result.config.clip_ratio,
                "desc_act": False,
                "model_file_base_name": "model",
            },
            f,
            indent=2,
        )
    written_files.append(config_path)

    total_size = sum(f.stat().st_size for f in written_files if f.suffix == ".safetensors")
    logger.info(
        f"Export complete: {len(written_files)} files, "
        f"{total_size / 1024**3:.2f} GB total, "
        f"compression ratio: {result.compression_ratio:.2f}x"
    )

    return written_files


def _create_shards(
    tensors: dict[str, torch.Tensor], max_bytes: int
) -> list[dict[str, torch.Tensor]]:
    """Split tensors into shards respecting max size."""
    shards: list[dict[str, torch.Tensor]] = [{}]
    current_size = 0

    for name, tensor in tensors.items():
        tensor_size = tensor.nelement() * tensor.element_size()

        if current_size + tensor_size > max_bytes and shards[-1]:
            shards.append({})
            current_size = 0

        shards[-1][name] = tensor
        current_size += tensor_size

    return shards
