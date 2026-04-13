# gemma4-quantizer

> **Native 3D tensor quantization for Google Gemma 4 MoE models**

Standard quantization tools (GPTQ, AWQ, bitsandbytes, modelopt) expect `nn.Linear` layers — they silently skip Gemma 4's fused 3D expert tensors `[num_experts, intermediate, hidden]`, leaving **~91% of the model unquantized**.

This library fixes that by quantizing fused 3D expert tensors **directly**, without unfusing them into individual expert modules.

## Quick Start

### Path 1: llama.cpp GGUF (Quickest)

The fastest path to running quantized Gemma 4 MoE locally:

```bash
# One-liner: build llama.cpp + convert + quantize
chmod +x scripts/quantize_with_llamacpp.sh
./scripts/quantize_with_llamacpp.sh google/gemma-4-26b-a4b-it Q4_K_M ./output

# Run inference
./llama.cpp/build/bin/llama-cli -m ./output/google_gemma-4-26b-a4b-it-Q4_K_M.gguf \
    -p "Hello, world!" -ngl 99
```

### Path 2: Python Library (More Control)

```bash
pip install -e ".[all]"
```

#### Analyze a checkpoint

```bash
# Fast header-only analysis (no weights loaded)
quantize-gemma4 analyze /path/to/gemma-4-26b-a4b-it --metadata-only --list-experts
```

```
============================================================
Gemma 4 MoE Checkpoint Analysis
============================================================
  Total parameters:     26,000,000,000
  Expert parameters:    23,660,000,000 (91.0%)
  Non-expert params:     2,340,000,000 (9.0%)
  Detected experts:                 128
  MoE layers:                        24
============================================================
```

#### Quantize with native 3D tensor support

```bash
# INT4 group quantization (recommended)
quantize-gemma4 quantize /path/to/model \
    -o ./quantized-gemma4 \
    --bits 4 \
    --method group \
    --group-size 128 \
    --validate

# Export as GGUF for llama.cpp
quantize-gemma4 quantize /path/to/model \
    -o ./gemma4.gguf \
    --bits 4 \
    --format gguf

# Mixed precision: 8-bit attention, 4-bit experts
quantize-gemma4 quantize /path/to/model \
    -o ./mixed-gemma4 \
    --bits 8 \
    --expert-bits 4
```

#### Python API

```python
from gemma4_quant import Gemma4Quantizer, QuantConfig, QuantMethod

# Configure
config = QuantConfig(
    bits=4,
    method=QuantMethod.GROUP,
    group_size=128,
    expert_bits=4,        # Can differ from attention bits
    clip_ratio=0.95,      # Clip outliers for better INT4
)

# Quantize
quantizer = Gemma4Quantizer(config)
result = quantizer.quantize_checkpoint("/path/to/gemma-4-26b-a4b-it/")

print(f"Compression: {result.compression_ratio:.1f}x")

# Export for vLLM (safetensors)
from gemma4_quant.exporters.safetensors_export import export_safetensors
export_safetensors(result, "./output/", pack_int4=True)

# Or for llama.cpp (GGUF)
from gemma4_quant.exporters.gguf_export import export_gguf
export_gguf(result, "./output/gemma4-q4.gguf")
```

#### Plug into existing frameworks

```python
# GPTQ: Patch AutoGPTQ to handle fused 3D experts
from gemma4_quant.plugins.gptq_plugin import patch_autogptq
patch_autogptq()

from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_pretrained("google/gemma-4-26b-a4b-it")
model.quantize(examples)  # Now correctly quantizes expert tensors

# AWQ: Same idea
from gemma4_quant.plugins.awq_plugin import patch_autoawq
patch_autoawq()
```

#### Validate quantization quality

```python
from gemma4_quant.validation import Validator

validator = Validator()
report = validator.validate(result, original_state_dict)
print(report.summary())
```

```
============================================================
Quantization Validation Report
============================================================
  Total tensors validated:  156
  Overall MSE:              0.00012345
  Overall Cosine Sim:       0.998765

  Grade distribution:
     A+:   42  ██████████████████████████████████████████
      A:   89  █████████████████████████████████████████████████████████████
      B:   25  █████████████████████████

  Worst experts (by cosine similarity):
    layers.12.experts.gate_up_proj expert[73]: cos=0.991234, mse=0.00098
============================================================
```

## How It Works

### The Problem
```
Standard quantizer:
  for module in model.modules():
      if isinstance(module, nn.Linear):   ← Only finds these
          quantize(module.weight)          ← 2D: [out, in]

  # Result: ~91% of Gemma 4 MoE params are SKIPPED
```

### Our Solution
```
gemma4-quantizer:
  for name, param in checkpoint.items():
      if param.ndim == 3 and is_expert(name):
          quantize_3d_expert(param)        ← 3D: [experts, intermediate, hidden]
                                             Per-expert, per-group scales
      elif param.ndim == 2:
          quantize_2d_linear(param)        ← Standard path

  # Result: 100% of quantizable params are handled
```

### Quantization Granularity for 3D Tensors
```
3D Tensor: [num_experts, intermediate_dim, hidden_dim]
                ↑                ↑                ↑
           Expert axis     Row axis          Column axis

Granularity: per-expert + per-group along hidden_dim
Scale shape: [num_experts, intermediate_dim, hidden_dim // group_size]

Each expert gets completely independent quantization scales.
```

## Architecture

```
gemma4-quantizer/
├── gemma4_quant/
│   ├── detector.py          # Classify tensors: 3D expert / 2D linear / norm
│   ├── quantizer.py         # Core quantization: absmax, group, asymmetric
│   ├── calibration.py       # Router-guided calibration for activation-aware quant
│   ├── validation.py        # Per-expert error analysis, perplexity benchmarks
│   ├── cli.py               # CLI: quantize-gemma4 analyze|quantize|validate
│   ├── exporters/
│   │   ├── safetensors_export.py   # vLLM/Marlin-compatible output
│   │   └── gguf_export.py          # llama.cpp-compatible output
│   └── plugins/
│       ├── gptq_plugin.py   # Monkey-patch AutoGPTQ
│       └── awq_plugin.py    # Monkey-patch AutoAWQ
├── scripts/
│   └── quantize_with_llamacpp.sh   # One-click llama.cpp pipeline
└── tests/
    └── test_quantizer.py    # Full test suite with synthetic MoE tensors
```

## Supported Configurations

| Method | Bits | Best For | Notes |
|--------|------|----------|-------|
| `group` | 4 | General use | Recommended default. group_size=128 |
| `group` | 8 | Quality-sensitive | Near-lossless |
| `absmax` | 8 | Speed | Fastest quantization |
| `absmax` | 4 | Quick & dirty | Lower quality than group |
| `asymmetric` | 4 | Skewed distributions | Extra zero-point overhead |
| Mixed | 8+4 | Best of both | 8-bit attention + 4-bit experts |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache 2.0
