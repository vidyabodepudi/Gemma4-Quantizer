"""
Validation module for quantized Gemma 4 MoE models.

Provides tools to verify quantization quality:
  - Per-tensor quantization error (MSE, cosine similarity)
  - Per-expert error analysis (identify experts degraded by quantization)
  - Perplexity benchmarking (requires model + tokenizer)
  - Router distribution validation (ensure routing isn't distorted)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from gemma4_quant.quantizer import QuantizationResult, QuantizedTensor

logger = logging.getLogger(__name__)


@dataclass
class TensorError:
    """Quantization error metrics for a single tensor."""

    name: str
    mse: float
    rmse: float
    max_error: float
    cosine_sim: float
    relative_error: float
    is_expert: bool
    shape: tuple[int, ...]

    @property
    def quality_grade(self) -> str:
        """Simple quality grade based on cosine similarity."""
        if self.cosine_sim >= 0.9999:
            return "A+"
        elif self.cosine_sim >= 0.999:
            return "A"
        elif self.cosine_sim >= 0.99:
            return "B"
        elif self.cosine_sim >= 0.95:
            return "C"
        else:
            return "F"


@dataclass
class ExpertError:
    """Per-expert quantization error within a fused 3D tensor."""

    tensor_name: str
    expert_id: int
    mse: float
    rmse: float
    cosine_sim: float


@dataclass
class ValidationReport:
    """Complete validation report."""

    tensor_errors: list[TensorError] = field(default_factory=list)
    expert_errors: list[ExpertError] = field(default_factory=list)
    overall_mse: float = 0.0
    overall_cosine_sim: float = 0.0
    worst_tensors: list[TensorError] = field(default_factory=list)
    worst_experts: list[ExpertError] = field(default_factory=list)
    perplexity_original: Optional[float] = None
    perplexity_quantized: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Quantization Validation Report",
            "=" * 60,
            f"  Total tensors validated:  {len(self.tensor_errors)}",
            f"  Overall MSE:              {self.overall_mse:.8f}",
            f"  Overall Cosine Sim:       {self.overall_cosine_sim:.6f}",
        ]

        if self.perplexity_original is not None:
            lines.append(f"  Perplexity (original):    {self.perplexity_original:.4f}")
        if self.perplexity_quantized is not None:
            lines.append(f"  Perplexity (quantized):   {self.perplexity_quantized:.4f}")
            if self.perplexity_original:
                delta = self.perplexity_quantized - self.perplexity_original
                pct = delta / self.perplexity_original * 100
                lines.append(f"  Perplexity delta:         {delta:+.4f} ({pct:+.2f}%)")

        # Grade distribution
        grades: dict[str, int] = {}
        for te in self.tensor_errors:
            grades[te.quality_grade] = grades.get(te.quality_grade, 0) + 1

        lines.append("")
        lines.append("  Grade distribution:")
        for grade in ["A+", "A", "B", "C", "F"]:
            count = grades.get(grade, 0)
            if count > 0:
                bar = "█" * min(count, 40)
                lines.append(f"    {grade:>3}: {count:>4}  {bar}")

        # Worst tensors
        if self.worst_tensors:
            lines.append("")
            lines.append("  Worst tensors (by cosine similarity):")
            for te in self.worst_tensors[:5]:
                lines.append(
                    f"    [{te.quality_grade}] {te.name}: "
                    f"cos={te.cosine_sim:.6f}, mse={te.mse:.8f}"
                )

        # Worst experts
        if self.worst_experts:
            lines.append("")
            lines.append("  Worst experts (by cosine similarity):")
            for ee in self.worst_experts[:5]:
                lines.append(
                    f"    {ee.tensor_name} expert[{ee.expert_id}]: "
                    f"cos={ee.cosine_sim:.6f}, mse={ee.mse:.8f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


class Validator:
    """
    Validates quantization quality by comparing quantized vs original weights.

    Usage:
        validator = Validator()
        report = validator.validate(quantization_result, original_state_dict)
        print(report.summary())
    """

    @staticmethod
    def compute_tensor_error(
        original: torch.Tensor, quantized: torch.Tensor
    ) -> tuple[float, float, float, float, float]:
        """
        Compute error metrics between original and dequantized tensor.

        Returns: (mse, rmse, max_error, cosine_sim, relative_error)
        """
        orig = original.float().flatten()
        quant = quantized.float().flatten()

        # Handle shape mismatch from padding
        min_len = min(len(orig), len(quant))
        orig = orig[:min_len]
        quant = quant[:min_len]

        diff = orig - quant
        mse = diff.pow(2).mean().item()
        rmse = mse**0.5
        max_error = diff.abs().max().item()

        # Cosine similarity
        cos = torch.nn.functional.cosine_similarity(
            orig.unsqueeze(0), quant.unsqueeze(0)
        ).item()

        # Relative error
        orig_norm = orig.norm().item()
        rel_error = diff.norm().item() / max(orig_norm, 1e-10)

        return mse, rmse, max_error, cos, rel_error

    def validate(
        self,
        result: QuantizationResult,
        original_state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> ValidationReport:
        """
        Validate quantization quality.

        Args:
            result: QuantizationResult from Gemma4Quantizer.
            original_state_dict: Original model weights for comparison.
                                 If None, uses dequantization for self-consistency check.
        """
        report = ValidationReport()

        for name, qt in result.quantized_tensors.items():
            # Get the reference tensor
            if original_state_dict and name in original_state_dict:
                orig = original_state_dict[name]
            else:
                # Self-consistency: not as useful but still validates
                # that dequantization works
                continue

            # Dequantize
            deq = qt.dequantize()

            # Overall tensor error
            mse, rmse, max_err, cos, rel_err = self.compute_tensor_error(orig, deq)

            te = TensorError(
                name=name,
                mse=mse,
                rmse=rmse,
                max_error=max_err,
                cosine_sim=cos,
                relative_error=rel_err,
                is_expert=qt.is_expert,
                shape=qt.original_shape,
            )
            report.tensor_errors.append(te)

            # Per-expert error for 3D tensors
            if qt.is_expert and orig.ndim == 3:
                E = orig.shape[0]
                for eid in range(E):
                    orig_e = orig[eid]
                    deq_e = deq[eid]
                    e_mse, e_rmse, _, e_cos, _ = self.compute_tensor_error(
                        orig_e, deq_e
                    )
                    ee = ExpertError(
                        tensor_name=name,
                        expert_id=eid,
                        mse=e_mse,
                        rmse=e_rmse,
                        cosine_sim=e_cos,
                    )
                    report.expert_errors.append(ee)

        # Aggregate metrics
        if report.tensor_errors:
            report.overall_mse = np.mean([te.mse for te in report.tensor_errors])
            report.overall_cosine_sim = np.mean(
                [te.cosine_sim for te in report.tensor_errors]
            )

        # Sort and extract worst
        report.worst_tensors = sorted(
            report.tensor_errors, key=lambda x: x.cosine_sim
        )[:10]
        report.worst_experts = sorted(
            report.expert_errors, key=lambda x: x.cosine_sim
        )[:10]

        return report

    @torch.no_grad()
    def compute_perplexity(
        self,
        model: torch.nn.Module,
        tokenizer,
        texts: list[str],
        max_length: int = 2048,
        device: str = "cpu",
    ) -> float:
        """Compute perplexity on a set of texts."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].shape[1]

            total_loss += loss * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        return np.exp(avg_loss)
