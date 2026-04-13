"""
CLI entry point for gemma4-quantizer.

Usage:
    # Analyze a checkpoint (no quantization, just detection)
    quantize-gemma4 analyze /path/to/model

    # Quantize to 4-bit with group quantization
    quantize-gemma4 quantize /path/to/model -o /path/to/output \\
        --bits 4 --method group --group-size 128

    # Quantize for llama.cpp GGUF
    quantize-gemma4 quantize /path/to/model -o /path/to/output.gguf \\
        --bits 4 --format gguf

    # Validate quantization quality
    quantize-gemma4 validate /path/to/quantized /path/to/original

    # Use with GPTQ plugin
    quantize-gemma4 patch-gptq
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_analyze(args):
    """Analyze a checkpoint to detect fused 3D expert tensors."""
    from gemma4_quant.detector import FusedExpertDetector

    detector = FusedExpertDetector()
    path = Path(args.model_path)

    print(f"\nAnalyzing: {path}")
    print()

    if path.suffix == ".safetensors" or path.is_dir():
        if args.metadata_only:
            analysis = detector.analyze_safetensors_metadata_only(path)
        else:
            analysis = detector.analyze_safetensors(path)
    else:
        print(f"Unsupported format: {path.suffix}")
        print("Supported: .safetensors files or directories containing them")
        sys.exit(1)

    print(analysis.summary())

    if args.list_experts:
        from gemma4_quant.detector import TensorKind

        print("\nFused 3D Expert Tensors:")
        print("-" * 80)
        for t in analysis.tensors:
            if t.kind == TensorKind.FUSED_EXPERT_3D:
                size_mb = t.size_bytes / 1024 / 1024
                print(
                    f"  {t.name}\n"
                    f"    shape={t.shape}, dtype={t.dtype}, "
                    f"experts={t.num_experts}, size={size_mb:.1f} MB"
                )


def cmd_quantize(args):
    """Quantize a Gemma 4 MoE model."""
    from gemma4_quant.quantizer import Gemma4Quantizer, QuantConfig, QuantMethod

    # Parse method
    method_map = {
        "absmax": QuantMethod.ABSMAX,
        "group": QuantMethod.GROUP,
        "asymmetric": QuantMethod.ASYMMETRIC,
    }
    method = method_map.get(args.method, QuantMethod.GROUP)

    config = QuantConfig(
        bits=args.bits,
        method=method,
        group_size=args.group_size,
        expert_bits=args.expert_bits,
        skip_embedding=not args.quantize_embeddings,
        clip_ratio=args.clip_ratio,
    )

    print(f"\nQuantization config:")
    print(f"  Bits:         {config.bits}")
    print(f"  Expert bits:  {config.expert_bits}")
    print(f"  Method:       {config.method.name}")
    print(f"  Group size:   {config.group_size}")
    print(f"  Clip ratio:   {config.clip_ratio}")
    print()

    quantizer = Gemma4Quantizer(config)
    result = quantizer.quantize_checkpoint(args.model_path)

    if result.errors:
        print(f"\n⚠ {len(result.errors)} errors during quantization:")
        for err in result.errors[:10]:
            print(f"  • {err}")

    print(f"\nCompression ratio: {result.compression_ratio:.2f}x")

    # Export
    output_path = Path(args.output)

    if args.format == "gguf" or output_path.suffix == ".gguf":
        from gemma4_quant.exporters.gguf_export import export_gguf

        out = export_gguf(result, output_path)
        print(f"\nGGUF exported: {out}")
    else:
        from gemma4_quant.exporters.safetensors_export import export_safetensors

        files = export_safetensors(result, output_path)
        print(f"\nSafeTensors exported: {len(files)} files to {output_path}/")

    # Optional validation
    if args.validate:
        print("\nRunning validation...")
        from gemma4_quant.validation import Validator

        validator = Validator()
        # Load original for comparison
        import torch

        if Path(args.model_path).is_dir():
            from safetensors.torch import load_file

            original = {}
            for sf in sorted(Path(args.model_path).glob("*.safetensors")):
                original.update(load_file(str(sf)))
        else:
            from safetensors.torch import load_file

            original = load_file(args.model_path)

        report = validator.validate(result, original)
        print(report.summary())


def cmd_validate(args):
    """Validate a quantized model against the original."""
    from gemma4_quant.validation import Validator

    print("Validation of pre-exported models is not yet implemented.")
    print("Use --validate flag during quantization instead.")


def cmd_patch_gptq(args):
    """Apply GPTQ patch for Gemma 4 MoE."""
    from gemma4_quant.plugins.gptq_plugin import patch_autogptq

    patch_autogptq()
    print("AutoGPTQ patched for Gemma 4 MoE fused 3D expert tensors.")
    print("You can now use AutoGPTQ as normal in this Python session.")


def cmd_patch_awq(args):
    """Apply AWQ patch for Gemma 4 MoE."""
    from gemma4_quant.plugins.awq_plugin import patch_autoawq

    patch_autoawq()
    print("AutoAWQ patched for Gemma 4 MoE fused 3D expert tensors.")


def main():
    parser = argparse.ArgumentParser(
        prog="quantize-gemma4",
        description="Native 3D tensor quantization for Google Gemma 4 MoE models",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Analyze ---
    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze a checkpoint for fused 3D expert tensors"
    )
    p_analyze.add_argument("model_path", help="Path to model or safetensors file")
    p_analyze.add_argument(
        "--list-experts", action="store_true", help="List all fused expert tensors"
    )
    p_analyze.add_argument(
        "--metadata-only",
        action="store_true",
        help="Read only safetensors headers (zero-copy, very fast)",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # --- Quantize ---
    p_quant = subparsers.add_parser("quantize", help="Quantize a Gemma 4 MoE model")
    p_quant.add_argument("model_path", help="Path to model or safetensors file")
    p_quant.add_argument(
        "-o", "--output", required=True, help="Output path (dir for safetensors, file for GGUF)"
    )
    p_quant.add_argument(
        "--bits", type=int, default=4, choices=[2, 3, 4, 8], help="Quantization bits (default: 4)"
    )
    p_quant.add_argument(
        "--expert-bits",
        type=int,
        default=None,
        choices=[2, 3, 4, 8],
        help="Override bits for expert tensors",
    )
    p_quant.add_argument(
        "--method",
        choices=["absmax", "group", "asymmetric"],
        default="group",
        help="Quantization method (default: group)",
    )
    p_quant.add_argument(
        "--group-size", type=int, default=128, help="Group size for group quantization (default: 128)"
    )
    p_quant.add_argument(
        "--clip-ratio", type=float, default=1.0, help="Clip ratio for outliers (default: 1.0)"
    )
    p_quant.add_argument(
        "--format",
        choices=["safetensors", "gguf"],
        default="safetensors",
        help="Output format (default: safetensors)",
    )
    p_quant.add_argument(
        "--quantize-embeddings",
        action="store_true",
        help="Also quantize embedding layer (usually not recommended)",
    )
    p_quant.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after quantization",
    )
    p_quant.set_defaults(func=cmd_quantize)

    # --- Validate ---
    p_val = subparsers.add_parser("validate", help="Validate a quantized model")
    p_val.add_argument("quantized_path", help="Path to quantized model")
    p_val.add_argument("original_path", help="Path to original model")
    p_val.set_defaults(func=cmd_validate)

    # --- Patch GPTQ ---
    p_gptq = subparsers.add_parser("patch-gptq", help="Patch AutoGPTQ for Gemma 4 MoE")
    p_gptq.set_defaults(func=cmd_patch_gptq)

    # --- Patch AWQ ---
    p_awq = subparsers.add_parser("patch-awq", help="Patch AutoAWQ for Gemma 4 MoE")
    p_awq.set_defaults(func=cmd_patch_awq)

    args = parser.parse_args()
    setup_logging(args.verbose)

    args.func(args)


if __name__ == "__main__":
    main()
