"""Competition score calculation and machine-readable report helpers."""
from __future__ import annotations


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_psnr(value: float) -> float:
    """20 points, linearly mapped from 15 dB (0) to 40 dB (20)."""
    return 20.0 * _clamp((value - 15.0) / (40.0 - 15.0), 0.0, 1.0)


def score_ssim(value: float) -> float:
    """20 points, linearly mapped from 0.70 (0) to 0.95 (20)."""
    return 20.0 * _clamp((value - 0.70) / (0.95 - 0.70), 0.0, 1.0)


def score_lpips(value: float) -> float:
    """40 points, linearly mapped from 0.25 (0) to 0.05 (40)."""
    return 40.0 * _clamp((0.25 - value) / (0.25 - 0.05), 0.0, 1.0)


def score_time_efficiency(module_b_seconds: float, budget_seconds: float) -> float:
    """20 points: full at <=10% budget, zero at or beyond the budget."""
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    ratio = module_b_seconds / budget_seconds
    if ratio <= 0.10:
        return 20.0
    if ratio >= 1.0:
        return 0.0
    return 20.0 * (1.0 - ratio) / 0.90


def build_report(*, scene, output, preset, checkpoint, module_b,
                 inference_seconds, module_b_seconds, t_iter_sec, metrics,
                 rendered_png_dir, ply_path):
    scores = {"psnr": None, "ssim": None, "lpips": None, "time_efficiency": None}
    metric_values = None
    if metrics and all(k in metrics for k in ("psnr", "ssim", "lpips")):
        metric_values = {k: float(metrics[k]) for k in ("psnr", "ssim", "lpips")}
        scores["psnr"] = round(score_psnr(metric_values["psnr"]), 4)
        scores["ssim"] = round(score_ssim(metric_values["ssim"]), 4)
        scores["lpips"] = round(score_lpips(metric_values["lpips"]), 4)

    budget_seconds = None
    ratio = None
    budget_status = "not_scored_missing_t_iter"
    if t_iter_sec is not None:
        budget_seconds = 30000.0 * float(t_iter_sec)
        ratio = module_b_seconds / budget_seconds
        scores["time_efficiency"] = round(score_time_efficiency(
            module_b_seconds, budget_seconds), 4)
        budget_status = "pass" if module_b_seconds < budget_seconds else "fail"

    total = None
    if all(scores[name] is not None for name in scores):
        total = round(sum(scores.values()), 4)

    return {
        "schema_version": 1,
        "scene": scene,
        "output": output,
        "model": {
            "preset": preset,
            "checkpoint": checkpoint,
            "resplat_modified": False,
        },
        "module_b": module_b,
        "timing": {
            "resplat_inference_seconds": round(float(inference_seconds), 6),
            "module_b_seconds": round(float(module_b_seconds), 6),
        },
        "time_budget": {
            "classic_3dgs_seconds_per_iteration": t_iter_sec,
            "allowed_seconds": round(budget_seconds, 6) if budget_seconds is not None else None,
            "used_fraction": round(ratio, 8) if ratio is not None else None,
            "strict_less_than_required": True,
            "status": budget_status,
        },
        "quality": {
            "evaluation_split": "all non-context COLMAP views selected by infer_colmap.py",
            "metrics": metric_values,
            "rendered_png_dir": rendered_png_dir,
            "refined_ply": ply_path,
        },
        "score": {
            "components": scores,
            "total": total,
            "maximum": 100,
            "weights": {"psnr": 20, "ssim": 20, "lpips": 40, "time_efficiency": 20},
        },
    }
