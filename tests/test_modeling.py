"""Meaningful estimator, metric, fold isolation and artifact round-trip checks."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from linkedin_analytics.modeling import (
    ModelingConfig, fit_glm, fit_interpretation, make_fold_assignments,
    modeling_rows, prediction_metrics, run_cross_validation, _robust_covariance,
)


def synthetic_postings(n=150):
    rng = np.random.default_rng(73)
    views = rng.integers(10, 200, n)
    easy = rng.integers(0, 2, n)
    role = rng.integers(0, 3, n)
    title = np.array(["Data analyst", "Software engineer", "Sales manager"])[role]
    descriptions = np.array(["python sql analytics team", "cloud software engineering development",
                             "sales marketing client account"])[role]
    mu = np.exp(.1 + .5 * np.log1p(views) + .2 * easy)
    days = rng.integers(20, 100, n).astype(float)
    days[:4] = np.nan
    return pd.DataFrame({
        "row_id": [f"job_{i}" for i in range(n)], "applies": rng.poisson(mu), "views": views,
        "title": title, "description": descriptions, "company_id": np.arange(n) // 3,
        "application_type": np.where(easy, "SimpleOnsiteApply", "OffsiteApply"),
        "formatted_work_type": "Full-time", "remote_allowed": "Unknown", "sponsored": "NotSponsored",
        "normalized_salary": rng.integers(50_000, 160_000, n),
        "legacy_days_open": days, "legacy_days_valid": np.isfinite(days),
    })


class ModelingTests(unittest.TestCase):
    def test_deviance_zeros_and_pooled_rmse(self):
        y, mu = np.array([0., 1., 5.]), np.array([1., 2., 4.])
        expected = 2 * (1 + np.log(.5) - 1 + 2 + 5 * np.log(5 / 4) - 5 + 4)
        metrics = prediction_metrics(y, mu)
        self.assertAlmostEqual(metrics["poisson_deviance_sum"], expected)
        self.assertAlmostEqual(metrics["mean_poisson_deviance"], expected / 3)
        self.assertAlmostEqual(metrics["rmse"], 1.)
        with self.assertRaises(ValueError):
            prediction_metrics(y, [1., 0., 4.])

    def test_offset_doubling_and_intercept_score_equation(self):
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(200, 2))
        exposure = rng.uniform(.5, 4, 200)
        y = rng.poisson(exposure * np.exp(.2 + .1 * matrix[:, 0]))
        cfg = ModelingConfig(n_threads=1)
        fit = fit_glm(matrix, y, np.log(exposure), alpha=0., config=cfg)
        pred = fit.predict(matrix, np.log(exposure))
        double = fit.predict(matrix, np.log(exposure) + np.log(2))
        np.testing.assert_allclose(double, pred * 2, rtol=1e-10)
        self.assertAlmostEqual(float(pred.sum()), float(y.sum()), delta=.01)

    def test_lasso_is_sparse_with_large_penalty(self):
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(100, 4))
        exposure = rng.uniform(1, 3, 100)
        y = rng.poisson(2 * exposure)
        fit = fit_glm(matrix, y, np.log(exposure), alpha=100., config=ModelingConfig(n_threads=1))
        np.testing.assert_allclose(fit.estimator.coef_, 0., atol=1e-9)
        np.testing.assert_allclose(fit.predict(matrix, np.log(exposure)), exposure * y.sum() / exposure.sum(), rtol=1e-5)

    def test_company_cluster_covariance_matches_independent_statsmodels(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(180, 3)) * np.array([.02, 1, 500])
        companies = np.repeat(np.arange(45), 4)
        mean = np.exp(.7 + matrix @ np.array([2., -.1, .0001]))
        y = rng.poisson(mean)
        fitted = sm.GLM(y, sm.add_constant(matrix), family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": companies, "use_correction": True})
        covariance, info = _robust_covariance(matrix, y, fitted.fittedvalues, companies)
        np.testing.assert_allclose(covariance, fitted.cov_params(), rtol=1e-7, atol=1e-9)
        self.assertEqual(info["covariance_type"], "company_cluster")
        self.assertLess(info["scaled_hessian_condition"], 10.)

    def test_missing_outcomes_and_views_are_excluded_not_imputed(self):
        frame = synthetic_postings(25)
        frame.loc[0, "applies"] = np.nan
        frame.loc[1, "views"] = np.nan
        eligible = modeling_rows(frame)
        self.assertEqual(len(eligible), 23)
        self.assertNotIn("job_0", eligible["row_id"].tolist())
        self.assertNotIn("job_1", eligible["row_id"].tolist())
        first, second = make_fold_assignments(eligible), make_fold_assignments(eligible)
        pd.testing.assert_frame_equal(first, second)

    def test_shared_oof_nested_isolation_and_saved_model_roundtrip(self):
        frame = synthetic_postings()
        cfg = ModelingConfig(n_svd=2, max_text_features=40, min_category_count=5,
                             min_text_df=1, alpha_grid=(.01, .1), max_boost_rounds=12,
                             early_stopping_rounds=4, n_threads=1, xgb_max_depth=2)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cross_validation(frame, cfg, tmp, progress=None)
            self.assertEqual(len(result["oof"]), len(frame))
            self.assertEqual(result["folds"]["row_id"].nunique(), len(frame))
            for column in ["pred_poisson", "pred_lasso_poisson", "pred_xgboost", "pred_poisson_no_svd"]:
                self.assertTrue(result["oof"][column].gt(0).all())
                self.assertTrue(np.isfinite(result["oof"][column]).all())
            self.assertEqual(result["oof"]["pred_poisson_legacy_offset"].isna().sum(), 4)
            for audit in result["feature_audit"]:
                test_ids = set(result["folds"].loc[result["folds"]["fold"] == audit["fold"], "row_id"])
                inner_train = set(audit["inner_train_ids"])
                inner_valid = set(audit["inner_validation_ids"])
                self.assertFalse(test_ids & inner_train)
                self.assertFalse(test_ids & inner_valid)
                self.assertFalse(inner_train & inner_valid)
                self.assertEqual(len(test_ids | inner_train | inner_valid), len(frame))
            fitted = fit_interpretation(frame, cfg, tmp, cv_result=result)
            loaded = joblib.load(Path(tmp) / "model_bundle.joblib")
            pd.testing.assert_frame_equal(fitted["bundle"].predict(frame.iloc[:5]), loaded.predict(frame.iloc[:5]))
            summary = json.loads((Path(tmp) / "final_fit_summary.json").read_text())
            self.assertEqual(summary["covariance_type"], "company_cluster")
            self.assertEqual(summary["n_rows"], len(frame))
            self.assertTrue((Path(tmp) / "cv_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
