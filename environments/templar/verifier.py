"""Verification functions for evaluation environment."""

import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class InnerStepsResult:
    """Expected return type from miner's inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


def validate_return_type(result) -> tuple[bool, str | None, InnerStepsResult | None]:
    """Validate that inner_steps returned correct type.

    Rejects proxy/lazy objects that override attribute-access or descriptor
    hooks to defer computation, which is a known exploit vector for faking MFU.
    """
    if isinstance(result, InnerStepsResult):
        return True, None, result

    _proxy_hooks = ("__getattr__", "__getattribute__", "__get__", "__set__", "__set_name__")
    result_type = type(result)
    for cls in result_type.__mro__:
        if cls is object:
            continue
        for hook in _proxy_hooks:
            if hook in cls.__dict__:
                return (
                    False,
                    f"Rejected proxy/lazy return type {result_type.__name__}: "
                    f"{hook} override detected in {cls.__name__} "
                    f"(deferred computation not allowed)",
                    None,
                )

    if all(hasattr(result, attr) for attr in ("final_logits", "total_tokens", "final_loss")):
        return (
            True,
            None,
            InnerStepsResult(
                final_logits=result.final_logits,
                total_tokens=result.total_tokens,
                final_loss=result.final_loss,
            ),
        )

    return False, f"Invalid return type from inner_steps: {type(result)}", None


def verify_trainable_params(
    model: torch.nn.Module,
    min_trainable_ratio: float = 1.0,
) -> tuple[bool, str | None, dict]:
    """Check that minimum % of params are trainable (prevents layer freezing)."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

    details = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "min_required": min_trainable_ratio,
    }

    if trainable_ratio < min_trainable_ratio:
        error = (
            f"Insufficient trainable params: {trainable_ratio:.1%} "
            f"({trainable_params:,}/{total_params:,}) - minimum {min_trainable_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Trainable params check ({trainable_ratio:.1%} >= {min_trainable_ratio:.0%})"
    )
    return True, None, details


def verify_params_changed(
    trained_state: dict,
    initial_state: dict,
    min_changed_ratio: float = 0.5,
    element_threshold: float = 1e-6,
) -> tuple[bool, str | None, dict]:
    """Verify that minimum % of individual parameter elements changed during training.

    Operates purely on CPU state dicts -- no model object or GPU memory needed.

    Args:
        trained_state: State dict after training (CPU tensors)
        initial_state: State dict before training (CPU tensors)
        min_changed_ratio: Minimum fraction of elements that must change
        element_threshold: Minimum absolute change for an element to count as "changed"
    """
    total_elements = 0
    changed_elements = 0

    for name, trained_val in trained_state.items():
        if name not in initial_state:
            continue
        initial_val = initial_state[name]
        if trained_val.shape != initial_val.shape:
            continue
        element_diffs = (trained_val.cpu().float() - initial_val.cpu().float()).abs()
        changed_mask = element_diffs > element_threshold

        total_elements += trained_val.numel()
        changed_elements += changed_mask.sum().item()

    changed_ratio = changed_elements / total_elements if total_elements > 0 else 0.0

    details = {
        "total_elements": total_elements,
        "changed_elements": int(changed_elements),
        "changed_ratio": changed_ratio,
        "min_required": min_changed_ratio,
        "element_threshold": element_threshold,
    }

    if changed_ratio < min_changed_ratio:
        error = (
            f"Insufficient parameter updates: {changed_ratio:.1%} "
            f"({int(changed_elements):,}/{total_elements:,} elements) "
            f"- minimum {min_changed_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Parameter changes check ({changed_ratio:.1%} >= {min_changed_ratio:.0%})"
    )
    return True, None, details


def verify_final_weights(
    candidate_state: dict,
    reference_final_state: dict,
    max_relative_error: float = 0.05,
) -> tuple[bool, str | None, dict]:
    """Verify miner's final weights match reference after full training.

    Operates purely on CPU state dicts -- no model object or GPU memory needed.
    Compares layer-by-layer to stay memory efficient.

    Args:
        candidate_state: Miner's state dict after training (CPU tensors)
        reference_final_state: Reference state dict after training (CPU tensors)
        max_relative_error: Maximum allowed |w_miner - w_ref| / |w_ref|

    Returns:
        Tuple of (success, error_message, details)
    """
    details: dict = {
        "relative_error_threshold": max_relative_error,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Final model weight comparison")
    logger.info("=" * 60)

    diff_norm_sq = 0.0
    ref_norm_sq = 0.0
    total_elements = 0
    shape_mismatched_layers = 0
    error_exceeded_layers = 0

    for name, cand_val in candidate_state.items():
        if name not in reference_final_state:
            continue

        ref_param = reference_final_state[name]
        if cand_val.shape != ref_param.shape:
            shape_mismatched_layers += 1
            continue
        diff = cand_val.cpu().float() - ref_param.cpu().float()

        layer_diff_sq = (diff * diff).sum().item()
        layer_ref_sq = (ref_param.cpu().float() * ref_param.cpu().float()).sum().item()

        diff_norm_sq += layer_diff_sq
        ref_norm_sq += layer_ref_sq
        total_elements += cand_val.numel()

        if layer_ref_sq > 0:
            layer_rel_error = (layer_diff_sq**0.5) / (layer_ref_sq**0.5)
            if not math.isfinite(layer_rel_error) or layer_rel_error > max_relative_error:
                error_exceeded_layers += 1

    mismatched_layers = shape_mismatched_layers + error_exceeded_layers

    ref_norm = ref_norm_sq**0.5
    diff_norm = diff_norm_sq**0.5

    if ref_norm > 0:
        relative_error = diff_norm / ref_norm
    else:
        relative_error = 0.0 if diff_norm == 0 else float("inf")

    details["relative_error"] = relative_error
    details["diff_norm"] = diff_norm
    details["ref_norm"] = ref_norm
    details["total_elements"] = total_elements

    logger.info(f"[CHECK] Weight relative error: {relative_error:.6f}")
    logger.info(f"   |w_miner - w_ref|: {diff_norm:.6f}")
    logger.info(f"   |w_ref|: {ref_norm:.6f}")
    logger.info(f"   Max allowed: {max_relative_error:.6f}")
    logger.info(f"   Total elements: {total_elements:,}")
    logger.info(f"   Shape mismatched layers: {shape_mismatched_layers}")
    logger.info(f"   Per-layer error exceeded layers: {error_exceeded_layers}")
    details["mismatched_layers"] = mismatched_layers
    details["shape_mismatched_layers"] = shape_mismatched_layers
    details["error_exceeded_layers"] = error_exceeded_layers

    if shape_mismatched_layers > 0:
        error = (
            f"{shape_mismatched_layers} layer(s) have shape mismatches — "
            f"model architecture was modified"
        )
        details["checks_failed"].append({"check": "shape_mismatch", "error": error})
        details["error_code"] = "shape_mismatch"
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Guard against NaN injection in weights
    if not math.isfinite(relative_error):
        error = (
            f"Weight relative error is non-finite ({relative_error}) - "
            "possible NaN injection detected"
        )
        details["checks_failed"].append({"check": "weight_nan_guard", "error": error})
        details["error_code"] = "weight_nan_injection"
        logger.error(f"[FAILED] {error}")
        return False, error, details

    if relative_error > max_relative_error:
        error = (
            f"Final weight relative error {relative_error:.6f} exceeds threshold "
            f"{max_relative_error:.6f} (|w_miner - w_ref| / |w_ref|)"
        )
        details["checks_failed"].append({"check": "weight_relative_error", "error": error})
        details["error_code"] = "weight_mismatch"
        logger.error(f"[FAILED] {error}")
        return False, error, details

    details["checks_passed"].append("weight_relative_error")
    logger.info("[PASSED] Final model weights match reference")

    logger.info("=" * 60)
    logger.info("VERIFICATION: WEIGHT CHECK PASSED")
    logger.info("=" * 60)

    return True, None, details


def verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
    expected_tokens: int,
    reference_final_state: dict | None = None,
    candidate_final_state: dict | None = None,
    max_loss_difference: float = 0.3,
    weight_relative_error_max: float = 0.008,
) -> tuple[bool, str | None, dict]:
    """Verify candidate outputs match reference.

    Verification checks (uniform across all parallelism strategies):
    1. Token count matches expected
    2. Loss is valid and similar to reference (small difference)
    3. Final weight verification (captures optimizer step correctness)

    Args:
        reference: Reference implementation results
        candidate: Miner's implementation results
        expected_tokens: Expected token count
        reference_final_state: Reference state dict after full training (CPU)
        candidate_final_state: Miner's state dict after training (CPU)
        max_loss_difference: Maximum allowed |candidate_loss - reference_loss|
        weight_relative_error_max: Max relative error for final weight comparison

    Returns:
        Tuple of (success, error_message, verification_details)
    """
    details = {
        "expected_tokens": expected_tokens,
        "candidate_tokens": candidate.total_tokens,
        "candidate_loss": candidate.final_loss,
        "reference_loss": reference.final_loss if reference else None,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Starting output verification")
    logger.info("=" * 60)

    # 1. Verify token count matches expected
    logger.info(
        f"[CHECK 1/3] Token count: expected={expected_tokens}, got={candidate.total_tokens}"
    )
    if candidate.total_tokens != expected_tokens:
        error = f"Token count mismatch: expected {expected_tokens}, got {candidate.total_tokens}"
        details["checks_failed"].append({"check": "token_count", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("token_count")
    logger.info("[PASSED] Token count matches")

    # 2. Verify loss is reasonable and similar to reference
    logger.info(f"[CHECK 2/3] Loss validity: candidate_loss={candidate.final_loss:.6f}")
    if candidate.final_loss != candidate.final_loss:  # NaN check
        error = "Loss is NaN"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if candidate.final_loss <= 0:
        error = f"Loss must be positive (cross-entropy > 0): got {candidate.final_loss:.4f}"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if candidate.final_loss > 100:
        error = f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Compare losses - check absolute difference
    if reference is None:
        error = "No reference result available for loss comparison"
        details["checks_failed"].append({"check": "loss_comparison", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    loss_difference = abs(candidate.final_loss - reference.final_loss)
    details["loss_difference"] = loss_difference
    details["reference_loss"] = reference.final_loss
    logger.info(
        f"   Reference loss: {reference.final_loss:.4f}, Candidate loss: {candidate.final_loss:.4f}"
    )
    logger.info(f"   Loss difference: {loss_difference:.4f} (max allowed: {max_loss_difference})")

    if loss_difference > max_loss_difference:
        error = (
            f"Loss difference too large: candidate={candidate.final_loss:.4f}, "
            f"reference={reference.final_loss:.4f}, "
            f"diff={loss_difference:.4f} > {max_loss_difference}"
        )
        details["checks_failed"].append({"check": "loss_comparison", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("loss_validity")
    logger.info("[PASSED] Loss is valid and similar to reference")

    # 3. Final weight verification (verifies optimizer step correctness)
    if reference_final_state is not None and candidate_final_state is not None:
        logger.info("[CHECK 3/3] Final weight verification")
        weight_ok, weight_error, weight_details = verify_final_weights(
            candidate_final_state,
            reference_final_state,
            max_relative_error=weight_relative_error_max,
        )
        details["weight_verification"] = weight_details

        if not weight_ok:
            details["checks_failed"].append({"check": "weight_verification", "error": weight_error})
            return False, weight_error, details
        details["checks_passed"].append("weight_verification")
    else:
        logger.warning("Skipping weight verification (no reference state available)")

    logger.info("=" * 60)
    logger.info("VERIFICATION: ALL CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details
