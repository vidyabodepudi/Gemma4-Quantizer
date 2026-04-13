"""
Unit tests for gemma4-quantizer.

Tests the core quantization engine on synthetic 3D tensors that simulate
the Gemma 4 MoE fused expert layout.
"""

import pytest
import torch

from gemma4_quant.detector import (
    CheckpointAnalysis,
    FusedExpertDetector,
    TensorKind,
)
from gemma4_quant.quantizer import (
    Gemma4Quantizer,
    QuantConfig,
    QuantMethod,
    QuantizedTensor,
)
from gemma4_quant.validation import Validator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gemma4_state_dict():
    """
    Create a synthetic state_dict that mimics Gemma 4 MoE tensor layout.
    This uses small dimensions to keep tests fast.
    """
    num_experts = 16
    hidden = 256
    intermediate = 512
    num_layers = 4
    vocab_size = 1000

    sd = {}

    # Embeddings
    sd["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden)

    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"

        # Attention
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden // 4, hidden)
        sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden // 4, hidden)
        sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)

        # Norms
        sd[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden)

        # MoE — fused 3D expert tensors (the key innovation)
        sd[f"{prefix}.block_sparse_moe.gate.weight"] = torch.randn(
            num_experts, hidden
        )
        sd[f"{prefix}.block_sparse_moe.experts.gate_up_proj"] = torch.randn(
            num_experts, 2 * intermediate, hidden
        )
        sd[f"{prefix}.block_sparse_moe.experts.down_proj"] = torch.randn(
            num_experts, hidden, intermediate
        )

    # Final norm and head
    sd["model.norm.weight"] = torch.randn(hidden)
    sd["lm_head.weight"] = torch.randn(vocab_size, hidden)

    return sd


@pytest.fixture
def detector():
    return FusedExpertDetector()


@pytest.fixture
def default_config():
    return QuantConfig(bits=4, method=QuantMethod.GROUP, group_size=64)


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


class TestFusedExpertDetector:

    def test_classify_3d_expert(self, detector):
        assert (
            detector.classify_tensor(
                "model.layers.0.block_sparse_moe.experts.gate_up_proj",
                (16, 1024, 256),
            )
            == TensorKind.FUSED_EXPERT_3D
        )

    def test_classify_2d_linear(self, detector):
        assert (
            detector.classify_tensor("model.layers.0.self_attn.q_proj.weight", (256, 256))
            == TensorKind.LINEAR_2D
        )

    def test_classify_embedding(self, detector):
        assert (
            detector.classify_tensor("model.embed_tokens.weight", (32000, 256))
            == TensorKind.EMBEDDING
        )

    def test_classify_norm(self, detector):
        assert (
            detector.classify_tensor("model.layers.0.input_layernorm.weight", (256,))
            == TensorKind.NORM_1D
        )

    def test_analyze_state_dict(self, detector, fake_gemma4_state_dict):
        analysis = detector.analyze_state_dict(fake_gemma4_state_dict)

        assert analysis.total_params > 0
        assert analysis.expert_params > 0
        assert analysis.num_experts_detected == 16
        assert analysis.num_moe_layers == 4
        assert analysis.expert_param_ratio > 0.5  # Experts should dominate

    def test_summary_output(self, detector, fake_gemma4_state_dict):
        analysis = detector.analyze_state_dict(fake_gemma4_state_dict)
        summary = analysis.summary()
        assert "Gemma 4 MoE Checkpoint Analysis" in summary
        assert "FUSED_EXPERT_3D" in summary


# ---------------------------------------------------------------------------
# Quantizer tests
# ---------------------------------------------------------------------------


class TestGemma4Quantizer:

    def test_3d_absmax_shapes(self):
        config = QuantConfig(bits=4, method=QuantMethod.ABSMAX)
        quantizer = Gemma4Quantizer(config)

        weight = torch.randn(16, 512, 256)
        q, scales = quantizer.quantize_3d_expert_absmax(weight, 4)

        assert q.shape == weight.shape
        assert scales.shape == (16, 512, 1)
        assert q.dtype == torch.int8

    def test_3d_group_shapes(self):
        config = QuantConfig(bits=4, method=QuantMethod.GROUP, group_size=64)
        quantizer = Gemma4Quantizer(config)

        weight = torch.randn(16, 512, 256)
        q, scales = quantizer.quantize_3d_expert_group(weight, 4, 64)

        assert q.shape == weight.shape
        assert scales.shape == (16, 512, 256 // 64)  # [E, I, num_groups]
        assert q.dtype == torch.int8

    def test_3d_asymmetric_shapes(self):
        config = QuantConfig(bits=4, method=QuantMethod.ASYMMETRIC, group_size=64)
        quantizer = Gemma4Quantizer(config)

        weight = torch.randn(16, 512, 256)
        q, scales, zeros = quantizer.quantize_3d_expert_asymmetric(weight, 4, 64)

        assert q.shape == weight.shape
        assert scales.shape == zeros.shape

    def test_quantize_value_range(self):
        config = QuantConfig(bits=4, method=QuantMethod.ABSMAX)
        quantizer = Gemma4Quantizer(config)

        weight = torch.randn(8, 128, 64) * 10
        q, _ = quantizer.quantize_3d_expert_absmax(weight, 4)

        # INT4 symmetric: values should be in [-7, 7]
        assert q.min().item() >= -7
        assert q.max().item() <= 7

    def test_2d_linear_quantization(self):
        config = QuantConfig(bits=8, method=QuantMethod.ABSMAX)
        quantizer = Gemma4Quantizer(config)

        weight = torch.randn(256, 256)
        q, scales = quantizer.quantize_2d_linear(weight, 8)

        assert q.shape == weight.shape
        assert q.dtype == torch.int8
        assert q.min().item() >= -127
        assert q.max().item() <= 127

    def test_quantize_state_dict(self, default_config, fake_gemma4_state_dict):
        quantizer = Gemma4Quantizer(default_config)
        result = quantizer.quantize_state_dict(
            fake_gemma4_state_dict, show_progress=False
        )

        # Should have quantized some tensors
        assert len(result.quantized_tensors) > 0
        # Norms should be passthrough
        assert len(result.passthrough_tensors) > 0
        # No errors
        assert len(result.errors) == 0

        # Check that expert tensors were quantized
        expert_quants = [
            qt for qt in result.quantized_tensors.values() if qt.is_expert
        ]
        assert len(expert_quants) > 0

    def test_dequantization_accuracy(self):
        """Test that quantize → dequantize preserves reasonable accuracy."""
        config = QuantConfig(bits=8, method=QuantMethod.ABSMAX)
        quantizer = Gemma4Quantizer(config)

        weight = torch.randn(8, 128, 64)
        q, scales = quantizer.quantize_3d_expert_absmax(weight, 8)

        # Dequantize
        deq = q.float() * scales.float()

        # INT8 should have high cosine similarity
        cos = torch.nn.functional.cosine_similarity(
            weight.flatten().unsqueeze(0),
            deq.flatten().unsqueeze(0),
        ).item()
        assert cos > 0.99, f"Cosine similarity too low: {cos}"

    def test_int4_dequantization(self):
        """Test INT4 dequantization accuracy (lower precision, more tolerance)."""
        config = QuantConfig(bits=4, method=QuantMethod.GROUP, group_size=32)
        quantizer = Gemma4Quantizer(config)

        weight = torch.randn(8, 128, 64)
        q, scales = quantizer.quantize_3d_expert_group(weight, 4, 32)

        # Dequantize
        E, I, H = weight.shape
        num_groups = H // 32
        q_reshaped = q.reshape(E, I, num_groups, 32)
        scales_reshaped = scales.unsqueeze(-1)  # [E, I, num_groups, 1]
        deq = (q_reshaped.float() * scales_reshaped.float()).reshape(E, I, H)

        cos = torch.nn.functional.cosine_similarity(
            weight.flatten().unsqueeze(0),
            deq.flatten().unsqueeze(0),
        ).item()
        assert cos > 0.95, f"Cosine similarity too low for INT4: {cos}"

    def test_skip_experts(self, fake_gemma4_state_dict):
        config = QuantConfig(bits=4, method=QuantMethod.GROUP, skip_experts=True)
        quantizer = Gemma4Quantizer(config)
        result = quantizer.quantize_state_dict(
            fake_gemma4_state_dict, show_progress=False
        )

        expert_quants = [
            qt for qt in result.quantized_tensors.values() if qt.is_expert
        ]
        assert len(expert_quants) == 0

    def test_different_expert_bits(self, fake_gemma4_state_dict):
        config = QuantConfig(bits=8, expert_bits=4, method=QuantMethod.GROUP)
        quantizer = Gemma4Quantizer(config)
        result = quantizer.quantize_state_dict(
            fake_gemma4_state_dict, show_progress=False
        )

        for qt in result.quantized_tensors.values():
            if qt.is_expert:
                assert qt.bits == 4
            else:
                assert qt.bits == 8


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidator:

    def test_compute_tensor_error(self):
        orig = torch.randn(100)
        noisy = orig + torch.randn(100) * 0.01

        mse, rmse, max_err, cos, rel_err = Validator.compute_tensor_error(
            orig, noisy
        )

        assert mse < 0.001
        assert cos > 0.99
        assert rel_err < 0.1

    def test_validate_result(self, default_config, fake_gemma4_state_dict):
        quantizer = Gemma4Quantizer(default_config)
        result = quantizer.quantize_state_dict(
            fake_gemma4_state_dict, show_progress=False
        )

        validator = Validator()
        report = validator.validate(result, fake_gemma4_state_dict)

        assert len(report.tensor_errors) > 0
        assert report.overall_cosine_sim > 0.9

        summary = report.summary()
        assert "Quantization Validation Report" in summary
        assert "Overall Cosine Sim" in summary


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestEndToEnd:

    def test_full_pipeline(self, fake_gemma4_state_dict, tmp_path):
        """Test: detect → quantize → validate full pipeline."""
        # 1. Detect
        detector = FusedExpertDetector()
        analysis = detector.analyze_state_dict(fake_gemma4_state_dict)
        assert analysis.expert_param_ratio > 0.5

        # 2. Quantize
        config = QuantConfig(bits=4, method=QuantMethod.GROUP, group_size=64)
        quantizer = Gemma4Quantizer(config)
        result = quantizer.quantize_state_dict(
            fake_gemma4_state_dict, show_progress=False
        )
        assert result.compression_ratio > 1.0

        # 3. Validate
        validator = Validator()
        report = validator.validate(result, fake_gemma4_state_dict)
        assert report.overall_cosine_sim > 0.9

        # 4. Check per-expert errors exist
        assert len(report.expert_errors) > 0

        # Print report
        print(report.summary())
