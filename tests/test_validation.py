"""Known-answer tests, independent of the modeling metric implementation."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from linkedin_analytics.validation import ResultValidationError, independent_metrics, verify_results

MODELS = ["constant_mean", "constant_exposure", "poisson", "lasso_poisson", "xgboost"]


def known_result_files(directory, *, legacy=False):
    # Hand calculation for y=(0,2), mu=(1,1): MAE=RMSE=1,
    # mean deviance=2*ln(2), sum deviance=4*ln(2).
    oof = pd.DataFrame({"row_id": ["001", "002"], "applies": [0, 2], "views": [1, 3], "fold": [1, 2]})
    summary, fold_metrics = [], []
    for model in MODELS:
        oof["pred_" + model] = [1.0, 1.0]
        summary.append(dict(model=model, n=2, mae=1, rmse=1,
                            mean_poisson_deviance=2 * math.log(2), poisson_deviance_sum=4 * math.log(2)))
        for fold, deviance in [(1, 2), (2, 4 * math.log(2) - 2)]:
            fold_metrics.append(dict(model=model, fold=fold, n=1, mae=1, rmse=1,
                                     mean_poisson_deviance=deviance, poisson_deviance_sum=deviance))
    if legacy:
        for model in ["poisson_legacy_offset", "poisson_legacy_matched_learned"]:
            oof["pred_" + model] = [float("nan"), 2.0]
            summary.append(dict(model=model, n=1, mae=0, rmse=0, mean_poisson_deviance=0, poisson_deviance_sum=0))
            fold_metrics.append(dict(model=model, fold=2, n=1, mae=0, rmse=0, mean_poisson_deviance=0, poisson_deviance_sum=0))
    oof.to_csv(directory / "cv_oof_predictions.csv", index=False)
    oof[["row_id", "fold"]].to_csv(directory / "fold_assignments.csv", index=False)
    pd.DataFrame(summary).to_csv(directory / "cv_summary.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(directory / "cv_fold_metrics.csv", index=False)
    (directory / "cv_run_summary.json").write_text(json.dumps({"config": {"n_splits": 2}, "population": {
        "modeling_rows": 2, "legacy_interval_rows": int(legacy), "target_zero_rows": 1}}), encoding="utf-8")


def test_scalar_metrics_match_hand_calculation_including_zero():
    metrics = independent_metrics([0, 2], [1, 1])
    assert metrics["mae"] == 1
    assert metrics["rmse"] == 1
    assert metrics["mean_poisson_deviance"] == pytest.approx(2 * math.log(2), abs=1e-14)
    assert metrics["poisson_deviance_sum"] == pytest.approx(4 * math.log(2), abs=1e-14)
    assert independent_metrics([2], [2])["mean_poisson_deviance"] == 0


def test_reconcile_saved_known_answer_and_matched_sensitivity(tmp_path):
    known_result_files(tmp_path, legacy=True)
    receipt = verify_results(tmp_path)
    assert receipt["status"] == "passed"
    assert receipt["n_rows"] == 2
    assert receipt["n_models"] == 7


@pytest.mark.parametrize("case", ["missing_prediction", "negative_prediction", "zero_prediction", "duplicate_row", "fold_change", "row_missing", "summary_tampered", "fold_metric_tampered"])
def test_detect_result_corruption(tmp_path, case):
    known_result_files(tmp_path)
    path = tmp_path / "cv_oof_predictions.csv"
    oof = pd.read_csv(path, dtype={"row_id": str})
    if case == "missing_prediction":
        oof.loc[0, "pred_poisson"] = float("nan")
    elif case == "negative_prediction":
        oof.loc[0, "pred_poisson"] = -1.0
    elif case == "zero_prediction":
        oof.loc[0, "pred_poisson"] = 0.0
    elif case == "duplicate_row":
        oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    elif case == "fold_change":
        oof.loc[0, "fold"] = 2
    elif case == "row_missing":
        oof = oof.iloc[[0]]
    elif case in {"summary_tampered", "fold_metric_tampered"}:
        filename = "cv_summary.csv" if case == "summary_tampered" else "cv_fold_metrics.csv"
        table = pd.read_csv(tmp_path / filename)
        table.loc[0, "mae"] = 99.0
        table.to_csv(tmp_path / filename, index=False)
    oof.to_csv(path, index=False)
    with pytest.raises(ResultValidationError):
        verify_results(tmp_path)


def test_legacy_pair_cannot_use_different_rows(tmp_path):
    known_result_files(tmp_path, legacy=True)
    path = tmp_path / "cv_oof_predictions.csv"
    oof = pd.read_csv(path, dtype={"row_id": str})
    oof["pred_poisson_legacy_matched_learned"] = [1.0, float("nan")]
    oof.to_csv(path, index=False)
    # Make that changed model's saved arithmetic internally consistent. The
    # failure must be unequal paired observations, not a metric discrepancy.
    model = "poisson_legacy_matched_learned"
    for name in ["cv_summary.csv", "cv_fold_metrics.csv"]:
        table = pd.read_csv(tmp_path / name)
        target = table.model.eq(model)
        table.loc[target, ["mae", "rmse"]] = 1.0
        table.loc[target, ["mean_poisson_deviance", "poisson_deviance_sum"]] = 2.0
        if "fold" in table:
            table.loc[target, "fold"] = 1
        table.to_csv(tmp_path / name, index=False)
    with pytest.raises(ResultValidationError, match="different observations"):
        verify_results(tmp_path)
