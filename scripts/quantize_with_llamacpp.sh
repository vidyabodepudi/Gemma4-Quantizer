#!/usr/bin/env bash
# =============================================================================
# Gemma 4 MoE — llama.cpp GGUF Quantization Quick Win Script
# =============================================================================
#
# This script automates:
#   1. Cloning and building llama.cpp
#   2. Converting Gemma 4 MoE from HuggingFace to GGUF
#   3. Quantizing to various bit-widths
#
# Prerequisites:
#   - cmake, make, git (brew install cmake)
#   - Python 3.10+ with pip
#   - ~60GB disk space for the full model + quantized versions
#
# Usage:
#   ./scripts/quantize_with_llamacpp.sh [MODEL_ID] [QUANT_TYPE] [OUTPUT_DIR]
#
# Examples:
#   # Default: Gemma 4 26B MoE, Q4_K_M quantization
#   ./scripts/quantize_with_llamacpp.sh
#
#   # Custom model and quant type
#   ./scripts/quantize_with_llamacpp.sh google/gemma-4-26b-a4b-it Q5_K_M ./my-models
#
#   # Just build llama.cpp (no conversion)
#   ./scripts/quantize_with_llamacpp.sh --build-only
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID="${1:-google/gemma-4-26b-a4b-it}"
QUANT_TYPE="${2:-Q4_K_M}"
OUTPUT_DIR="${3:-./quantized-models}"
LLAMA_CPP_DIR="./llama.cpp"
BUILD_ONLY="${1:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 0: Check prerequisites
# ---------------------------------------------------------------------------

check_prereqs() {
    info "Checking prerequisites..."

    local missing=()

    command -v git >/dev/null 2>&1    || missing+=("git")
    command -v cmake >/dev/null 2>&1  || missing+=("cmake")
    command -v make >/dev/null 2>&1   || missing+=("make")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")

    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing prerequisites: ${missing[*]}\n  Install with: brew install ${missing[*]}"
    fi

    ok "All prerequisites found"
}

# ---------------------------------------------------------------------------
# Step 1: Clone and build llama.cpp
# ---------------------------------------------------------------------------

build_llama_cpp() {
    info "Setting up llama.cpp..."

    if [ -d "$LLAMA_CPP_DIR" ]; then
        info "llama.cpp directory exists, pulling latest..."
        cd "$LLAMA_CPP_DIR"
        git pull --ff-only || warn "Could not pull latest (may be on a tag)"
        cd ..
    else
        info "Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    fi

    info "Building llama.cpp..."
    cd "$LLAMA_CPP_DIR"

    # Clean build
    rm -rf build
    mkdir build
    cd build

    # Configure with Metal support (macOS) or CUDA if available
    if [ "$(uname)" = "Darwin" ]; then
        info "Configuring with Metal (Apple GPU) support..."
        cmake .. -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
    elif command -v nvcc >/dev/null 2>&1; then
        info "Configuring with CUDA support..."
        cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    else
        info "Configuring CPU-only build..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
    fi

    cmake --build . --config Release -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

    cd ../..
    ok "llama.cpp built successfully"
}

# ---------------------------------------------------------------------------
# Step 2: Install Python dependencies for conversion
# ---------------------------------------------------------------------------

install_python_deps() {
    info "Installing Python dependencies..."

    pip3 install --user --upgrade \
        torch \
        transformers \
        safetensors \
        sentencepiece \
        protobuf \
        numpy \
        tqdm \
        huggingface_hub \
        gguf \
        2>&1 | tail -5

    ok "Python dependencies installed"
}

# ---------------------------------------------------------------------------
# Step 3: Download model from HuggingFace
# ---------------------------------------------------------------------------

download_model() {
    local model_dir="./models/${MODEL_ID//\//_}"

    if [ -d "$model_dir" ] && ls "$model_dir"/*.safetensors 1>/dev/null 2>&1; then
        info "Model already downloaded: $model_dir"
        echo "$model_dir"
        return
    fi

    info "Downloading model: $MODEL_ID"
    info "This may take a while for large models..."

    python3 -c "
from huggingface_hub import snapshot_download
import os

model_dir = '$model_dir'
os.makedirs(model_dir, exist_ok=True)

snapshot_download(
    repo_id='$MODEL_ID',
    local_dir=model_dir,
    ignore_patterns=['*.bin', '*.pt', 'original/*'],
)
print(f'Downloaded to: {model_dir}')
"
    ok "Model downloaded: $model_dir"
    echo "$model_dir"
}

# ---------------------------------------------------------------------------
# Step 4: Convert to GGUF (F16)
# ---------------------------------------------------------------------------

convert_to_gguf() {
    local model_dir="$1"
    local gguf_f16="${OUTPUT_DIR}/${MODEL_ID//\//_}-f16.gguf"

    mkdir -p "$OUTPUT_DIR"

    if [ -f "$gguf_f16" ]; then
        info "F16 GGUF already exists: $gguf_f16"
        echo "$gguf_f16"
        return
    fi

    info "Converting to GGUF (F16)..."
    info "This handles the fused 3D expert tensors natively."

    python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
        "$model_dir" \
        --outfile "$gguf_f16" \
        --outtype f16

    ok "Converted to GGUF: $gguf_f16"
    echo "$gguf_f16"
}

# ---------------------------------------------------------------------------
# Step 5: Quantize
# ---------------------------------------------------------------------------

quantize_model() {
    local gguf_f16="$1"
    local gguf_quant="${OUTPUT_DIR}/${MODEL_ID//\//_}-${QUANT_TYPE}.gguf"

    if [ -f "$gguf_quant" ]; then
        info "Quantized GGUF already exists: $gguf_quant"
        echo "$gguf_quant"
        return
    fi

    info "Quantizing: $QUANT_TYPE"
    info "  Input:  $gguf_f16"
    info "  Output: $gguf_quant"

    "${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
        "$gguf_f16" \
        "$gguf_quant" \
        "$QUANT_TYPE"

    ok "Quantized model saved: $gguf_quant"

    # Show file sizes
    local f16_size=$(du -h "$gguf_f16" | cut -f1)
    local quant_size=$(du -h "$gguf_quant" | cut -f1)
    info "Size comparison:"
    info "  F16:     $f16_size"
    info "  $QUANT_TYPE:  $quant_size"

    echo "$gguf_quant"
}

# ---------------------------------------------------------------------------
# Step 6: Quick test inference
# ---------------------------------------------------------------------------

test_inference() {
    local gguf_path="$1"

    info "Running quick test inference..."

    "${LLAMA_CPP_DIR}/build/bin/llama-cli" \
        -m "$gguf_path" \
        -p "Hello, I am a Gemma 4 model and" \
        -n 50 \
        --temp 0.7 \
        -ngl 99 \
        --no-display-prompt \
        2>/dev/null || warn "Test inference failed (may need more VRAM)"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo ""
    echo "=============================================="
    echo " Gemma 4 MoE → GGUF Quantization Pipeline"
    echo "=============================================="
    echo ""
    echo "  Model:      $MODEL_ID"
    echo "  Quant Type: $QUANT_TYPE"
    echo "  Output:     $OUTPUT_DIR"
    echo ""

    check_prereqs

    if [ "$BUILD_ONLY" = "--build-only" ]; then
        build_llama_cpp
        ok "Build complete. Run again without --build-only to convert a model."
        exit 0
    fi

    build_llama_cpp
    install_python_deps

    local model_dir
    model_dir=$(download_model)

    local gguf_f16
    gguf_f16=$(convert_to_gguf "$model_dir")

    local gguf_quant
    gguf_quant=$(quantize_model "$gguf_f16")

    echo ""
    echo "=============================================="
    echo " Done!"
    echo "=============================================="
    echo ""
    echo " Quantized model: $gguf_quant"
    echo ""
    echo " Run with llama.cpp:"
    echo "   ${LLAMA_CPP_DIR}/build/bin/llama-cli \\"
    echo "     -m $gguf_quant \\"
    echo "     -p 'Your prompt here' \\"
    echo "     -ngl 99  # offload all layers to GPU"
    echo ""
    echo " Available quant types (re-run with different type):"
    echo "   Q2_K    — 2-bit (smallest, lowest quality)"
    echo "   Q3_K_M  — 3-bit medium"
    echo "   Q4_K_M  — 4-bit medium (recommended balance) ← DEFAULT"
    echo "   Q5_K_M  — 5-bit medium (good quality)"
    echo "   Q6_K    — 6-bit (near-FP16 quality)"
    echo "   Q8_0    — 8-bit (highest quality, largest)"
    echo ""

    read -p "Run a quick test inference? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_inference "$gguf_quant"
    fi
}

main "$@"
