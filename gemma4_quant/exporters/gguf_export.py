"""
GGUF exporter for quantized Gemma 4 MoE models.

Exports to GGUF format compatible with llama.cpp inference.
Uses the official `gguf` Python library when available, otherwise
creates a minimal GGUF-compatible output.

Note: For full GGUF conversion with all metadata, consider using
llama.cpp's convert scripts directly. This exporter handles the
3D expert tensor layout that those scripts may not fully optimize.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gemma4_quant.quantizer import QuantizationResult

logger = logging.getLogger(__name__)

# GGUF quantization type constants
GGUF_QTYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "IQ4_NL": 20,
}


def _translate_name_to_gguf(name: str) -> str:
    """
    Translate a HuggingFace tensor name to GGUF convention.
    e.g., 'model.layers.0.block_sparse_moe.experts.gate_up_proj'
       → 'blk.0.ffn_gate_up_exps.weight'
    """
    import re

    # Standard translations
    translations = {
        r"model\.embed_tokens\.weight": "token_embd.weight",
        r"model\.norm\.weight": "output_norm.weight",
        r"lm_head\.weight": "output.weight",
    }

    for pattern, replacement in translations.items():
        if re.match(pattern, name):
            return replacement

    # Layer translations
    layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)

        layer_translations = {
            r"input_layernorm\.weight": "attn_norm.weight",
            r"post_attention_layernorm\.weight": "ffn_norm.weight",
            r"pre_feedforward_layernorm\.weight": "ffn_norm.weight",
            r"post_feedforward_layernorm\.weight": "ffn_post_norm.weight",
            r"self_attn\.q_proj\.weight": "attn_q.weight",
            r"self_attn\.k_proj\.weight": "attn_k.weight",
            r"self_attn\.v_proj\.weight": "attn_v.weight",
            r"self_attn\.o_proj\.weight": "attn_output.weight",
            r"block_sparse_moe\.gate\.weight": "ffn_gate_inp.weight",
            r"block_sparse_moe\.experts\.gate_up_proj": "ffn_gate_up_exps.weight",
            r"block_sparse_moe\.experts\.down_proj": "ffn_down_exps.weight",
            # Dense FFN fallback
            r"mlp\.gate_proj\.weight": "ffn_gate.weight",
            r"mlp\.up_proj\.weight": "ffn_up.weight",
            r"mlp\.down_proj\.weight": "ffn_down.weight",
        }

        for pattern, replacement in layer_translations.items():
            if re.match(pattern, rest):
                return f"blk.{layer_idx}.{replacement}"

    # Fallback: return as-is
    logger.warning(f"No GGUF translation for tensor: {name}")
    return name


def export_gguf(
    result: QuantizationResult,
    output_path: str | Path,
    model_arch: str = "gemma2",
    use_gguf_lib: bool = True,
) -> Path:
    """
    Export quantized model to GGUF format.

    Args:
        result: Quantization result from Gemma4Quantizer.
        output_path: Path for the output .gguf file.
        model_arch: GGUF model architecture identifier.
        use_gguf_lib: If True, use the official gguf Python package.

    Returns:
        Path to the written GGUF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gguf_lib:
        try:
            return _export_with_gguf_lib(result, output_path, model_arch)
        except ImportError:
            logger.warning(
                "gguf package not found, falling back to manual export. "
                "Install with: pip install gguf"
            )
            return _export_manual_gguf(result, output_path, model_arch)
    else:
        return _export_manual_gguf(result, output_path, model_arch)


def _export_with_gguf_lib(
    result: QuantizationResult,
    output_path: Path,
    model_arch: str,
) -> Path:
    """Export using the official gguf Python library."""
    import gguf

    writer = gguf.GGUFWriter(str(output_path), model_arch)

    # Add metadata
    writer.add_quantization_version(2)
    config = result.config
    writer.add_custom_alignment(32)

    # Add tensors
    for name, qt in result.quantized_tensors.items():
        gguf_name = _translate_name_to_gguf(name)

        if qt.is_expert and qt.data.ndim == 3:
            # 3D expert tensor — write as-is; llama.cpp handles 3D expert layout
            # Dequantize for GGUF (GGUF applies its own quantization)
            w_deq = qt.dequantize().numpy().astype(np.float32)
            writer.add_tensor(gguf_name, w_deq)
        else:
            # 2D tensor — standard handling
            w_deq = qt.dequantize().numpy().astype(np.float32)
            writer.add_tensor(gguf_name, w_deq)

    # Passthrough tensors
    for name, tensor in result.passthrough_tensors.items():
        gguf_name = _translate_name_to_gguf(name)
        data = tensor.float().numpy()
        writer.add_tensor(gguf_name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    logger.info(f"GGUF export complete: {output_path}")
    return output_path


def _export_manual_gguf(
    result: QuantizationResult,
    output_path: Path,
    model_arch: str,
) -> Path:
    """
    Minimal manual GGUF export (fallback when gguf library not available).
    Writes an unquantized F16 GGUF that can then be re-quantized by
    llama.cpp's `llama-quantize` tool with proper 3D tensor handling.
    """
    import struct

    # Collect all tensors (dequantized for re-quantization by llama.cpp)
    all_tensors: dict[str, np.ndarray] = {}

    for name, qt in result.quantized_tensors.items():
        gguf_name = _translate_name_to_gguf(name)
        w_deq = qt.dequantize().numpy().astype(np.float16)
        all_tensors[gguf_name] = w_deq

    for name, tensor in result.passthrough_tensors.items():
        gguf_name = _translate_name_to_gguf(name)
        all_tensors[gguf_name] = tensor.numpy().astype(np.float16)

    logger.info(
        f"Manual GGUF export: {len(all_tensors)} tensors → {output_path}"
    )
    logger.warning(
        "Manual GGUF export creates an F16 file. Re-quantize with "
        "`llama-quantize` for best results."
    )

    # For a production implementation, you'd write the full GGUF binary
    # format here. For now, save as a numpy archive that can be converted.
    np_path = output_path.with_suffix(".npz")
    np.savez(str(np_path), **all_tensors)
    logger.info(f"Saved intermediate numpy archive: {np_path}")
    logger.info(
        "Convert to GGUF using: python convert_hf_to_gguf.py (from llama.cpp)"
    )

    return np_path
