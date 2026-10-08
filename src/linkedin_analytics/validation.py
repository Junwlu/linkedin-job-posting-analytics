"""Independent, read-only reconciliation of saved out-of-fold evidence.

This module does not import modeling.py or reuse its metric implementation.
Passing artifact checks does not itself prove that a model was trained without
leakage; training-only transforms and split isolation need separate tests.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_MODELS = {"constant_mean", "constant_exposure", "poisson", "lasso_poisson", "xgboost"}
LEGACY_MODELS = {"poisson_legacy_offset", "poisson_legacy_matched_learned"}
ALLOWED_MODELS = PRIMARY_MODELS | LEGACY_MODELS | {"poisson_no_svd", "poisson_views_offset"}
METRIC_COLUMNS = ("mae", "rmse", "mean_poisson_deviance", "poisson_deviance_sum")


class ResultValidationError(ValueError):
    """Saved evidence fails an independent reconciliation check."""


def independent_metrics(y, predicted) -> dict:
    """Compute pooled metrics by scalar arithmetic, including observed zeros."""
    outcomes = [float(value) for value in y]
    means = [float(value) for value in predicted]
    if not outcomes or len(outcomes) != len(means):
        raise ResultValidationError("Metric arrays must have the same nonzero length")
    if any(not math.isfinite(value) or value < 0 for value in outcomes):
        raise ResultValidationError("Outcomes must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in means):
        raise ResultValidationError("Predictions must be finite and strictly positive")
    residuals = [observed - mean for observed, mean in zip(outcomes, means)]
    terms = []
    for observed, mean in zip(outcomes, means):
        if observed == 0:
            term = 2.0 * mean
        else:
            term = 2.0 * (observed * (math.log(observed) - math.log(mean)) - observed + mean)
        if term < -1e-8:
            raise ResultValidationError("Poisson deviance is materially negative")
        terms.append(max(term, 0.0))
    count = len(outcomes)
    result = {
        "n": count,
        "mae": math.fsum(abs(value) for value in residuals) / count,
        "rmse": math.sqrt(math.fsum(value * value for value in residuals) / count),
        "mean_poisson_deviance": math.fsum(terms) / count,
        "poisson_deviance_sum": math.fsum(terms),
    }
    if any(not math.isfinite(result[key]) for key in METRIC_COLUMNS):
        raise ResultValidationError("Recomputed metrics are not finite")
    return result


def _require(condition, message):
    if not condition:
        raise ResultValidationError(message)


def _numeric(series: pd.Series, label: str, *, allow_missing=False) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ResultValidationError(f"{label} contains non-numeric values") from exc
    _require(not np.isinf(values).any(), f"{label} contains infinite values")
    if not allow_missing:
        _require(np.isfinite(values).all(), f"{label} contains missing/nonfinite values")
    return values


def _check_metrics(record: pd.Series, expected: dict, context: str, *, rtol, atol):
    for key in ("n", *METRIC_COLUMNS):
        try:
            value = float(record[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultValidationError(f"{context}: missing/non-numeric {key}") from exc
        _require(math.isfinite(value), f"{context}: nonfinite {key}")
        if key == "n":
            _require(value == expected[key], f"{context}: n differs from OOF coverage")
        else:
            _require(math.isclose(value, expected[key], rel_tol=rtol, abs_tol=atol),
                     f"{context}: {key}={value} differs from independently recomputed {expected[key]}")


def verify_results(output_dir: Path | str, *, rtol=1e-9, atol=1e-8) -> dict:
    """Verify coverage, folds, and saved global/per-fold metrics; never modify files.

    Required files are cv_oof_predictions.csv, cv_summary.csv,
    cv_fold_metrics.csv, fold_assignments.csv and cv_run_summary.json.
    Raises ResultValidationError on any discrepancy and returns a small receipt
    otherwise. Both legacy columns may be missing together on ineligible rows.
    """
    directory = Path(output_dir)
    names = ["cv_oof_predictions.csv", "cv_summary.csv", "cv_fold_metrics.csv",
             "fold_assignments.csv", "cv_run_summary.json"]
    for name in names:
        _require((directory / name).is_file(), f"Required result file missing: {name}")
    oof = pd.read_csv(directory / names[0], dtype={"row_id": str})
    summary = pd.read_csv(directory / names[1])
    fold_metrics = pd.read_csv(directory / names[2])
    assignments = pd.read_csv(directory / names[3], dtype={"row_id": str})
    run = json.loads((directory / names[4]).read_text(encoding="utf-8"))
    _require({"row_id", "fold", "applies", "views"}.issubset(oof), "OOF identity/target columns missing")
    _require({"row_id", "fold"}.issubset(assignments), "Fold identity columns missing")
    _require(len(oof) > 0, "OOF file has no observations")
    for label, frame in [("OOF", oof), ("fold assignments", assignments)]:
        _require(frame.row_id.notna().all() and frame.row_id.str.strip().ne("").all(), f"{label} has missing row_id")
        _require(not frame.row_id.duplicated().any(), f"{label} has duplicate row_id")
    _require(set(oof.row_id) == set(assignments.row_id), "OOF row_id coverage differs from fold assignments")
    folds = _numeric(oof.fold, "OOF fold")
    reference_folds = _numeric(assignments.set_index("row_id").loc[oof.row_id, "fold"], "assigned fold")
    _require(np.all((folds >= 1) & (folds == np.floor(folds))), "Fold labels must be positive integers")
    _require(np.array_equal(folds, reference_folds), "OOF folds differ from independently saved assignments")
    population = run.get("population", {})
    _require(population.get("modeling_rows") == len(oof), "Run population differs from OOF row coverage")
    split_count = run.get("config", {}).get("n_splits")
    if split_count is not None:
        _require(set(folds) == set(range(1, int(split_count) + 1)), "Configured outer folds are missing or unexpected")
    target = _numeric(oof.applies, "applies")
    views = _numeric(oof.views, "views")
    _require(np.all((target >= 0) & (target == np.floor(target))), "OOF target is not a nonnegative integer count")
    _require(np.all(views >= 0), "OOF views must be nonnegative")
    if "target_zero_rows" in population:
        _require(population["target_zero_rows"] == int((target == 0).sum()), "Run zero-target count differs from OOF")

    models = {column.removeprefix("pred_") for column in oof if column.startswith("pred_")}
    _require(PRIMARY_MODELS.issubset(models), "A primary model/baseline prediction column is missing")
    _require(models.issubset(ALLOWED_MODELS), "An unknown prediction/model column is present")
    variants = run.get("config", {}).get("model_variants", [])
    for variant, model in [("no_svd", "poisson_no_svd"), ("views_offset", "poisson_views_offset")]:
        if variant in variants:
            _require(model in models, f"Predeclared sensitivity prediction is missing: {model}")
    if "legacy_offset" in variants and split_count is not None:
        if population.get("legacy_interval_rows", 0) >= int(split_count) * 10:
            _require(LEGACY_MODELS.issubset(models), "Predeclared eligible legacy comparison is missing")
    _require({"model", "n", *METRIC_COLUMNS}.issubset(summary), "Summary metric columns missing")
    _require({"model", "fold", "n", *METRIC_COLUMNS}.issubset(fold_metrics), "Fold metric columns missing")
    _require(not summary.model.duplicated().any(), "Summary has duplicate model rows")
    _require(set(summary.model) == models, "Summary models differ from OOF prediction columns")
    _require(not fold_metrics.duplicated(["model", "fold"]).any(), "Fold metrics have duplicate model/fold rows")
    _numeric(fold_metrics.fold, "metric fold")
    present_legacy = models & LEGACY_MODELS
    _require(not present_legacy or present_legacy == LEGACY_MODELS, "Legacy comparison pair is incomplete")
    masks, metrics, expected_fold_pairs = {}, {}, set()
    for model in sorted(models):
        means = _numeric(oof["pred_" + model], model, allow_missing=model in LEGACY_MODELS)
        keep = np.isfinite(means)
        if model not in LEGACY_MODELS:
            _require(keep.all(), f"{model} has missing OOF observations")
        _require(np.all(means[keep] > 0), f"{model} has zero/negative predictions")
        masks[model] = keep
        expected = independent_metrics(target[keep], means[keep])
        _check_metrics(summary.loc[summary.model.eq(model)].iloc[0], expected, model, rtol=rtol, atol=atol)
        metrics[model] = expected
        for fold in sorted(set(folds[keep])):
            same_fold = keep & (folds == fold)
            expected_fold_pairs.add((model, int(fold)))
            record = fold_metrics.loc[fold_metrics.model.eq(model) & fold_metrics.fold.eq(fold)]
            _require(len(record) == 1, f"{model}/fold {fold}: missing or duplicate fold metric")
            expected_fold = independent_metrics(target[same_fold], means[same_fold])
            _check_metrics(record.iloc[0], expected_fold, f"{model}/fold {fold}", rtol=rtol, atol=atol)
    actual_fold_pairs = set(zip(fold_metrics.model, fold_metrics.fold))
    _require(actual_fold_pairs == expected_fold_pairs, "Unexpected or missing model/fold metrics")
    if present_legacy:
        a, b = sorted(LEGACY_MODELS)
        _require(np.array_equal(masks[a], masks[b]), "Legacy sensitivity pair uses different observations")
        _require(int(masks[a].sum()) == population.get("legacy_interval_rows"), "Legacy sample count differs from run population")
    receipt = {
        "status": "passed", "n_rows": len(oof), "n_folds": len(set(folds)),
        "n_models": len(models), "recomputed_metrics": metrics,
        "rtol": rtol, "atol": atol,
        "oof_sha256": hashlib.sha256((directory / names[0]).read_bytes()).hexdigest(),
        "checks": ["unique complete row_id coverage", "saved outer fold alignment",
                   "common primary-model observations", "finite positive predictions",
                   "independent pooled and fold metrics", "matched legacy sensitivity observations"],
        "scope": "Saved-result arithmetic and coverage; preprocessing isolation is checked separately.",
    }
    return receipt
