"""Shared-fold, nested-holdout comparisons of conditional application counts.

The primary models learn the coefficient/function of log(1 + views). Offsets
are confined to explicitly labeled sensitivity analyses. Outer held-out rows
never fit preprocessing or select hyperparameters.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import xlogy
from scipy.stats import norm
from sklearn.model_selection import KFold, train_test_split

from .features import FeatureBuilder, FeatureConfig


@dataclass
class ModelingConfig:
    seed: int = 20251201
    n_splits: int = 5
    inner_fraction: float = .2
    n_svd: int = 15
    max_text_features: int = 1500
    min_category_count: int = 20
    min_text_df: int = 2
    alpha_grid: tuple[float, ...] = (.001, .01, .1, 1.)
    max_boost_rounds: int = 800
    early_stopping_rounds: int = 40
    n_threads: int = 3
    glm_max_iter: int = 200
    glm_gradient_tol: float = 1e-5
    xgb_max_depth: int = 4
    xgb_eta: float = .05
    model_variants: tuple[str, ...] = ("main", "no_svd", "views_offset", "legacy_offset")


def _config(config=None) -> ModelingConfig:
    if config is None:
        return ModelingConfig()
    if isinstance(config, ModelingConfig):
        return config
    if isinstance(config, dict):
        valid = {item.name for item in fields(ModelingConfig)}
        unknown = set(config) - valid
        if unknown:
            raise ValueError(f"Unknown modeling config fields: {sorted(unknown)}")
        return ModelingConfig(**config)
    raise TypeError("config must be a ModelingConfig or dictionary.")


def _builder(config: ModelingConfig, *, include_svd=True, include_log_views=True) -> FeatureBuilder:
    return FeatureBuilder(FeatureConfig(
        n_svd=config.n_svd, max_text_features=config.max_text_features,
        min_category_count=config.min_category_count, min_text_df=config.min_text_df,
        random_state=config.seed, include_svd=include_svd, include_log_views=include_log_views,
    ))


def modeling_rows(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Select the common measured-outcome/measured-views population explicitly."""
    needed = {"row_id", "applies", "views"}
    if not needed.issubset(clean_df.columns):
        raise ValueError(f"Missing modeling columns: {needed - set(clean_df.columns)}")
    frame = clean_df.copy()
    y = pd.to_numeric(frame["applies"], errors="coerce").to_numpy(float)
    views = pd.to_numeric(frame["views"], errors="coerce").to_numpy(float)
    keep = np.isfinite(y) & (y >= 0) & np.isclose(y, np.round(y))
    keep &= np.isfinite(views) & (views >= 0)
    frame = frame.loc[keep].reset_index(drop=True)
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise ValueError("row_id must be nonmissing and unique before modeling.")
    frame["row_id"] = frame["row_id"].astype(str)
    if len(frame) < 10:
        raise ValueError("Not enough eligible posting rows for modeling.")
    return frame


def prediction_metrics(y, mu) -> dict:
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    if y.shape != mu.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("Metrics require same-length nonempty one-dimensional arrays.")
    if not np.all(np.isfinite(y) & (y >= 0)):
        raise ValueError("Targets must be finite and nonnegative.")
    if not np.all(np.isfinite(mu) & (mu > 0)):
        raise ValueError("Predicted means must be finite and strictly positive; no silent clipping.")
    # xlogy(0, 0 / mu) = 0, so observed zeros are handled exactly.
    unit = 2 * (xlogy(y, y / mu) - y + mu)
    if np.min(unit) < -1e-8:
        raise ArithmeticError("Poisson deviance unexpectedly negative.")
    unit = np.maximum(unit, 0.)  # Only remove floating-point roundoff near equality.
    return {
        "n": int(len(y)), "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "mean_poisson_deviance": float(np.mean(unit)),
        "poisson_deviance_sum": float(np.sum(unit)),
    }


def make_fold_assignments(frame: pd.DataFrame, config=None) -> pd.DataFrame:
    cfg = _config(config)
    folds = np.full(len(frame), -1, dtype=int)
    for fold, (_, test) in enumerate(KFold(cfg.n_splits, shuffle=True, random_state=cfg.seed).split(frame), 1):
        folds[test] = fold
    return pd.DataFrame({"row_id": frame["row_id"].astype(str).to_numpy(), "fold": folds})


def _resolve_folds(frame: pd.DataFrame, config, fold_ids=None) -> pd.DataFrame:
    if fold_ids is None:
        return make_fold_assignments(frame, config)
    if isinstance(fold_ids, pd.DataFrame):
        if fold_ids["row_id"].astype(str).duplicated().any():
            raise ValueError("Duplicate row_id in fold assignments.")
        mapping = fold_ids.assign(row_id=fold_ids["row_id"].astype(str)).set_index("row_id")["fold"]
        values = frame["row_id"].map(mapping)
        if values.isna().any():
            raise ValueError("Missing fold assignment for eligible rows.")
        values = values.to_numpy(int)
    else:
        values = np.asarray(fold_ids, dtype=int)
        if len(values) != len(frame):
            raise ValueError("fold_ids length differs from the eligible population.")
    if len(np.unique(values)) != config.n_splits:
        raise ValueError("Fold assignments do not have the configured number of folds.")
    return pd.DataFrame({"row_id": frame["row_id"].to_numpy(), "fold": values})


def _independent_columns(matrix: np.ndarray) -> np.ndarray:
    """Keep a full-rank slope design with an unpenalized intercept forced in."""
    if matrix.shape[1] == 0:
        return np.array([], dtype=int)
    centered = matrix - np.mean(matrix, axis=0)
    scales = np.std(centered, axis=0)
    nonconstant = np.flatnonzero(scales > 1e-12)
    if not len(nonconstant):
        return nonconstant
    normalized = centered[:, nonconstant] / scales[nonconstant]
    _, r, pivot = qr(normalized, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    rank = int(np.sum(diagonal > max(normalized.shape) * np.finfo(float).eps * diagonal.max()))
    return np.sort(nonconstant[pivot[:rank]])


@dataclass
class GLMFit:
    estimator: object | None
    selected_indices: np.ndarray
    feature_names: list[str]
    dropped_aliases: list[str]
    intercept_only_rate: float | None = None
    fit_warnings: list[str] | None = None

    def predict(self, matrix, offset):
        if self.estimator is None:
            return np.exp(np.asarray(offset, float)) * self.intercept_only_rate
        return np.asarray(self.estimator.predict(
            np.asfortranarray(matrix[:, self.selected_indices]), offset=np.asarray(offset, float)
        ), dtype=float)


def fit_glm(matrix, y, offset, feature_names=None, *, alpha=0., config=None) -> GLMFit:
    """Fit a genuine Poisson GLM/Lasso with offset at fitting and prediction."""
    from glum import GeneralizedLinearRegressor
    from threadpoolctl import threadpool_limits

    cfg = _config(config)
    matrix = np.asarray(matrix, dtype=float)
    y, offset = np.asarray(y, float), np.asarray(offset, float)
    names = list(feature_names or [f"x{i}" for i in range(matrix.shape[1])])
    selected = _independent_columns(matrix) if alpha == 0 else np.flatnonzero(np.std(matrix, axis=0) > 1e-12)
    dropped = [name for i, name in enumerate(names) if i not in set(selected)]
    if len(selected) == 0:
        rate = max(float(np.sum(y) / np.exp(offset).sum()), np.finfo(float).tiny)
        return GLMFit(None, selected, [], dropped, intercept_only_rate=rate, fit_warnings=[])
    estimator = GeneralizedLinearRegressor(
        family="poisson", link="log", alpha=float(alpha), l1_ratio=1. if alpha > 0 else 0.,
        fit_intercept=True, scale_predictors=alpha > 0,
        solver="irls-cd" if alpha > 0 else "irls-ls",
        max_iter=cfg.glm_max_iter, gradient_tol=cfg.glm_gradient_tol,
        random_state=cfg.seed,
    )
    with threadpool_limits(limits=cfg.n_threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(np.asfortranarray(matrix[:, selected]), y, offset=offset)
    messages = [str(item.message) for item in caught]
    if any("converg" in message.lower() for message in messages):
        raise RuntimeError("GLM did not converge: " + "; ".join(messages))
    fit = GLMFit(estimator, selected, [names[i] for i in selected], dropped, fit_warnings=messages)
    prediction_metrics(y, fit.predict(matrix, offset))
    return fit


def _xgb_params(config):
    return {
        "objective": "count:poisson", "eval_metric": "poisson-nloglik",
        "max_depth": config.xgb_max_depth, "eta": config.xgb_eta,
        "subsample": .8, "colsample_bytree": .8, "min_child_weight": 5,
        "tree_method": "hist", "nthread": config.n_threads, "seed": config.seed,
    }


def _dmatrix(matrix, *, y=None, offset=None, names=None, n_threads=3):
    import xgboost as xgb
    return xgb.DMatrix(np.asarray(matrix, dtype=np.float32), label=y,
                       base_margin=np.asarray(offset, dtype=np.float32),
                       feature_names=names, nthread=n_threads)


def _tune_main(train: pd.DataFrame, config: ModelingConfig, fold: int):
    import xgboost as xgb

    inner_train, inner_valid = train_test_split(
        np.arange(len(train)), test_size=config.inner_fraction, random_state=config.seed + int(fold),
    )
    training, validation = train.iloc[inner_train], train.iloc[inner_valid]
    builder = _builder(config)
    x_train, x_valid = builder.fit_transform(training), builder.transform(validation)
    y_train, y_valid = training["applies"].to_numpy(float), validation["applies"].to_numpy(float)
    off_train, off_valid = np.zeros(len(training)), np.zeros(len(validation))
    candidate_rows = []
    for alpha in config.alpha_grid:
        fitted = fit_glm(x_train, y_train, off_train, builder.feature_names_, alpha=alpha, config=config)
        metrics = prediction_metrics(y_valid, fitted.predict(x_valid, off_valid))
        candidate_rows.append({"fold": fold, "alpha": float(alpha), **metrics})
    # Ties choose the larger penalty for the more parsimonious model.
    best = min(candidate_rows, key=lambda row: (row["mean_poisson_deviance"], -row["alpha"]))
    dtrain = _dmatrix(x_train, y=y_train, offset=off_train, names=builder.feature_names_, n_threads=config.n_threads)
    dvalid = _dmatrix(x_valid, y=y_valid, offset=off_valid, names=builder.feature_names_, n_threads=config.n_threads)
    booster = xgb.train(
        _xgb_params(config), dtrain, num_boost_round=config.max_boost_rounds,
        evals=[(dvalid, "inner_validation")], early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )
    rounds = int(booster.best_iteration) + 1
    return {
        "alpha": best["alpha"], "rounds": rounds, "candidates": candidate_rows,
        "inner_train_ids": training["row_id"].tolist(), "inner_validation_ids": validation["row_id"].tolist(),
        "inner_feature_audit": builder.diagnostics(validation),
        "inner_best_mean_deviance": best["mean_poisson_deviance"],
        "xgb_inner_best_nloglik": float(booster.best_score),
    }


def _offset(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "learned_views":
        return np.zeros(len(frame))
    offset = np.log1p(frame["views"].to_numpy(float))
    if mode == "legacy_offset":
        days = pd.to_numeric(frame["legacy_days_open"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(days) & (days > 0)):
            raise ValueError("Legacy offset requires valid positive planned intervals.")
        offset += np.log(days)
    elif mode != "views_offset":
        raise ValueError(f"Unknown offset mode {mode!r}")
    return offset


def _save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_cross_validation(clean_df, config=None, output_dir=None, fold_ids=None,
                         progress: Callable[[str], None] | None = print) -> dict:
    """Run all predeclared comparisons and save inspectable out-of-fold evidence.

    Returned DataFrames: oof, fold_metrics, summary, tuning, tuning_candidates,
    folds. Other entries: config, feature_audit, population, elapsed_seconds.
    Missing sensitivity predictions mean an ineligible interval, never zero.
    """
    import xgboost as xgb

    started = time.monotonic()
    cfg = _config(config)
    frame = modeling_rows(clean_df)
    folds = _resolve_folds(frame, cfg, fold_ids)
    oof = frame[["row_id", "applies", "views"]].copy()
    oof["fold"] = folds["fold"].to_numpy()
    models = ["constant_mean", "constant_exposure", "poisson", "lasso_poisson", "xgboost"]
    if "no_svd" in cfg.model_variants:
        models.append("poisson_no_svd")
    if "views_offset" in cfg.model_variants:
        models.append("poisson_views_offset")
    legacy_mask = np.zeros(len(frame), dtype=bool)
    if "legacy_offset" in cfg.model_variants and "legacy_days_open" in frame:
        days = pd.to_numeric(frame["legacy_days_open"], errors="coerce").to_numpy(float)
        legacy_mask = np.isfinite(days) & (days > 0)
        if "legacy_days_valid" in frame:
            legacy_mask &= frame["legacy_days_valid"].fillna(False).to_numpy(bool)
        if legacy_mask.sum() >= cfg.n_splits * 10:
            models.extend(["poisson_legacy_offset", "poisson_legacy_matched_learned"])
    for name in models:
        oof["pred_" + name] = np.nan
    metrics_rows, tuning_rows, candidate_rows, audits = [], [], [], []
    for fold in sorted(folds["fold"].unique()):
        train_idx = np.flatnonzero(folds["fold"].to_numpy() != fold)
        test_idx = np.flatnonzero(folds["fold"].to_numpy() == fold)
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        if progress:
            progress(f"Fold {fold}/{cfg.n_splits}: inner tuning on {len(train):,} training postings.")
        tuning = _tune_main(train, cfg, int(fold))
        candidate_rows.extend(tuning.pop("candidates"))
        builder = _builder(cfg)
        x_train, x_test = builder.fit_transform(train), builder.transform(test)
        y_train, y_test = train["applies"].to_numpy(float), test["applies"].to_numpy(float)
        off_train, off_test = np.zeros(len(train)), np.zeros(len(test))
        names = builder.feature_names_
        fold_preds = {
            "constant_mean": np.full(len(test), max(float(y_train.mean()), np.finfo(float).tiny)),
            "constant_exposure": (test["views"].to_numpy(float) + 1.) *
                max(float(y_train.sum() / (train["views"].to_numpy(float) + 1.).sum()), np.finfo(float).tiny),
        }
        poisson = fit_glm(x_train, y_train, off_train, names, config=cfg)
        fold_preds["poisson"] = poisson.predict(x_test, off_test)
        lasso = fit_glm(x_train, y_train, off_train, names, alpha=tuning["alpha"], config=cfg)
        fold_preds["lasso_poisson"] = lasso.predict(x_test, off_test)
        dtrain = _dmatrix(x_train, y=y_train, offset=off_train, names=names, n_threads=cfg.n_threads)
        dtest = _dmatrix(x_test, offset=off_test, names=names, n_threads=cfg.n_threads)
        booster = xgb.train(_xgb_params(cfg), dtrain, num_boost_round=tuning["rounds"], verbose_eval=False)
        fold_preds["xgboost"] = np.asarray(booster.predict(dtest), dtype=float)
        if "poisson_no_svd" in models:
            idx = np.array([i for i, name in enumerate(names) if not name.startswith("desc_svd_")])
            fitted = fit_glm(x_train[:, idx], y_train, off_train, [names[i] for i in idx], config=cfg)
            fold_preds["poisson_no_svd"] = fitted.predict(x_test[:, idx], off_test)
        if "poisson_views_offset" in models:
            idx = np.array([i for i, name in enumerate(names) if name != "log1p_views"])
            fitted = fit_glm(x_train[:, idx], y_train, _offset(train, "views_offset"), [names[i] for i in idx], config=cfg)
            fold_preds["poisson_views_offset"] = fitted.predict(x_test[:, idx], _offset(test, "views_offset"))
        for model_name, prediction in fold_preds.items():
            metrics_rows.append({"model": model_name, "fold": int(fold), **prediction_metrics(y_test, prediction)})
            oof.loc[test_idx, "pred_" + model_name] = prediction
        legacy_audit = None
        if "poisson_legacy_offset" in models:
            legacy_train_idx = train_idx[legacy_mask[train_idx]]
            legacy_test_idx = test_idx[legacy_mask[test_idx]]
            legacy_train, legacy_test = frame.iloc[legacy_train_idx], frame.iloc[legacy_test_idx]
            if len(legacy_train) < 10 or len(legacy_test) < 1:
                raise ValueError("Too few valid legacy-interval rows in an outer fold.")
            legacy_builder = _builder(cfg)
            lx_train, lx_test = legacy_builder.fit_transform(legacy_train), legacy_builder.transform(legacy_test)
            ly_train, ly_test = legacy_train["applies"].to_numpy(float), legacy_test["applies"].to_numpy(float)
            legacy_names = legacy_builder.feature_names_
            for label, mode in [("poisson_legacy_offset", "legacy_offset"),
                                ("poisson_legacy_matched_learned", "learned_views")]:
                idx = np.array([i for i, name in enumerate(legacy_names)
                                if mode == "learned_views" or name != "log1p_views"])
                fitted = fit_glm(lx_train[:, idx], ly_train, _offset(legacy_train, mode),
                                 [legacy_names[i] for i in idx], config=cfg)
                prediction = fitted.predict(lx_test[:, idx], _offset(legacy_test, mode))
                oof.loc[legacy_test_idx, "pred_" + label] = prediction
                metrics_rows.append({"model": label, "fold": int(fold), **prediction_metrics(ly_test, prediction)})
            legacy_audit = legacy_builder.diagnostics(legacy_test)
        audits.append({
            "fold": int(fold), "outer": builder.diagnostics(test),
            "inner_train_ids": tuning.pop("inner_train_ids"),
            "inner_validation_ids": tuning.pop("inner_validation_ids"),
            "inner": tuning.pop("inner_feature_audit"),
            "poisson_dropped_aliases": poisson.dropped_aliases,
            "poisson_fit_warnings": poisson.fit_warnings,
            "lasso_fit_warnings": lasso.fit_warnings,
            "legacy": legacy_audit,
        })
        tuning_rows.append({"fold": int(fold), "n_train": len(train), "n_test": len(test),
                            "n_features": x_train.shape[1], **tuning})
        if progress:
            progress(f"Fold {fold} complete: alpha={tuning['alpha']:g}, XGBoost rounds={tuning['rounds']}.")
    summary = []
    for model in models:
        valid = oof["pred_" + model].notna()
        expected = legacy_mask if model.startswith("poisson_legacy_") else np.ones(len(frame), bool)
        if not np.array_equal(valid.to_numpy(), expected):
            raise AssertionError(f"OOF coverage mismatch for {model}.")
        summary.append({"model": model, **prediction_metrics(
            oof.loc[valid, "applies"].to_numpy(float), oof.loc[valid, "pred_" + model].to_numpy(float))})
    result = {
        "oof": oof, "fold_metrics": pd.DataFrame(metrics_rows), "summary": pd.DataFrame(summary),
        "tuning": pd.DataFrame(tuning_rows), "tuning_candidates": pd.DataFrame(candidate_rows),
        "folds": folds, "feature_audit": audits, "config": asdict(cfg),
        "population": {"input_rows": len(clean_df), "modeling_rows": len(frame),
                       "legacy_interval_rows": int(legacy_mask.sum()),
                       "target_zero_rows": int((frame["applies"] == 0).sum())},
        "elapsed_seconds": time.monotonic() - started,
    }
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for key, filename in {
            "oof": "cv_oof_predictions.csv", "fold_metrics": "cv_fold_metrics.csv",
            "summary": "cv_summary.csv", "tuning": "cv_tuning.csv",
            "tuning_candidates": "cv_tuning_candidates.csv", "folds": "fold_assignments.csv",
        }.items():
            result[key].to_csv(directory / filename, index=False)
        _save_json(directory / "feature_audit.json", audits)
        _save_json(directory / "cv_run_summary.json", {
            "config": result["config"], "population": result["population"],
            "elapsed_seconds": result["elapsed_seconds"],
            "estimand": "Conditional recorded application counts among measured positive outcomes and views.",
            "main_offset": "zero; log1p(views) is a learned predictor",
            "metric_aggregation": "All metrics pooled over one out-of-fold prediction per eligible row.",
        })
    return result


def _robust_covariance(matrix, y, mu, clusters=None):
    # Invert in a centered/scaled parameterization, then transform covariance
    # back to the original units. Do not silently truncate Hessian directions.
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    if np.any(scales <= 1e-12):
        raise ArithmeticError("Constant slope remains in the inference design.")
    design = np.column_stack([np.ones(len(matrix)), (matrix - means) / scales])
    n, p = design.shape
    hessian = design.T @ (mu[:, None] * design)
    condition = float(np.linalg.cond(hessian))
    if n <= p or not np.isfinite(condition) or condition > 1e12:
        raise ArithmeticError("Ill-conditioned inference Hessian; coefficient intervals withheld.")
    bread = np.linalg.solve(hessian, np.eye(p))
    score = design * (y - mu)[:, None]
    kind, groups = "HC1", None
    if clusters is not None:
        _, codes = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
        groups = int(codes.max()) + 1
        if groups > 1:
            aggregated = np.zeros((groups, p))
            np.add.at(aggregated, codes, score)
            meat = aggregated.T @ aggregated
            correction = groups / (groups - 1) * (n - 1) / (n - p)
            kind = "company_cluster"
        else:
            meat, correction = score.T @ score, n / (n - p)
    else:
        meat, correction = score.T @ score, n / (n - p)
    covariance_scaled = correction * (bread @ meat @ bread)
    jacobian = np.eye(p)
    jacobian[0, 1:] = -means / scales
    jacobian[1:, 1:] = np.diag(1 / scales)
    covariance = jacobian @ covariance_scaled @ jacobian.T
    covariance = (covariance + covariance.T) / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or eigenvalues.min() < -1e-8 * max(1., eigenvalues.max()):
        raise ArithmeticError("Robust covariance failed the finite/positive-semidefinite check.")
    return covariance, {"covariance_type": kind, "n_clusters": groups,
                        "design_rank": int(np.linalg.matrix_rank(design)),
                        "n_parameters": p, "scaled_hessian_condition": condition}


@dataclass
class ModelBundle:
    """Trusted local model artifact; do not load joblib files from unknown senders."""
    features: FeatureBuilder
    poisson: GLMFit
    lasso: GLMFit
    xgboost: object
    config: dict

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self.features.transform(frame)
        offset = np.zeros(len(frame))
        dmatrix = _dmatrix(matrix, offset=offset, names=self.features.feature_names_,
                           n_threads=self.config["n_threads"])
        return pd.DataFrame({
            "poisson": self.poisson.predict(matrix, offset),
            "lasso_poisson": self.lasso.predict(matrix, offset),
            "xgboost": self.xgboost.predict(dmatrix),
        }, index=frame.index)


def fit_interpretation(clean_df, config=None, output_dir=None, cv_result=None) -> dict:
    """Fit full-data associations and save fitted transforms, separate from CV.

    If cv_result is omitted, cv_tuning.csv must already exist in output_dir.
    Final hyperparameters are medians of inner-selected outer-fold values;
    no outer loss is used to select them. Coefficient CIs are descriptive.
    """
    import joblib
    import xgboost as xgb

    cfg = _config(config)
    frame = modeling_rows(clean_df)
    directory = Path(output_dir) if output_dir is not None else None
    if cv_result is not None:
        tuning = cv_result["tuning"]
    elif directory is not None and (directory / "cv_tuning.csv").exists():
        tuning = pd.read_csv(directory / "cv_tuning.csv")
    else:
        raise ValueError("Provide cv_result or an output directory with cv_tuning.csv.")
    alpha = float(np.exp(np.median(np.log(tuning["alpha"].to_numpy(float)))))
    rounds = max(1, int(np.round(np.median(tuning["rounds"].to_numpy(float)))))
    builder = _builder(cfg)
    matrix = builder.fit_transform(frame)
    names = builder.feature_names_
    y, offset = frame["applies"].to_numpy(float), np.zeros(len(frame))
    poisson = fit_glm(matrix, y, offset, names, config=cfg)
    lasso = fit_glm(matrix, y, offset, names, alpha=alpha, config=cfg)
    booster = xgb.train(_xgb_params(cfg), _dmatrix(matrix, y=y, offset=offset, names=names, n_threads=cfg.n_threads),
                        num_boost_round=rounds, verbose_eval=False)
    mu = poisson.predict(matrix, offset)
    selected = poisson.selected_indices
    if poisson.estimator is None:
        beta = np.array([np.log(poisson.intercept_only_rate)])
    else:
        beta = np.r_[poisson.estimator.intercept_, poisson.estimator.coef_]
    clusters = None
    if "company_id" in frame:
        raw = frame["company_id"].astype("string")
        clusters = np.array([f"missing_{i}" if pd.isna(value) or str(value) in ("", "0", "nan")
                             else str(value) for i, value in enumerate(raw)], dtype=str)
    covariance, covariance_info = _robust_covariance(matrix[:, selected], y, mu, clusters)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    z = np.divide(beta, errors, out=np.full_like(beta, np.nan), where=errors > 0)
    with np.errstate(over="ignore"):
        irr, low, high = np.exp(beta), np.exp(beta - 1.96 * errors), np.exp(beta + 1.96 * errors)
    metadata = {row["name"]: row for row in builder.feature_metadata_}
    coefficient_names = ["Intercept"] + poisson.feature_names
    coefficients = pd.DataFrame({
        "feature": coefficient_names, "coefficient": beta, "robust_std_error": errors,
        "irr": irr, "ci_lower": low, "ci_upper": high, "p_value": 2 * norm.sf(np.abs(z)),
        "covariance_type": covariance_info["covariance_type"],
        "unit": ["baseline expected count" if name == "Intercept" else metadata[name]["unit"] for name in coefficient_names],
    })
    coefficients["ci_excludes_one"] = (coefficients["ci_lower"] > 1) | (coefficients["ci_upper"] < 1)
    lasso_coefs = np.array([]) if lasso.estimator is None else lasso.estimator.coef_
    lasso_table = pd.DataFrame({"feature": lasso.feature_names, "coefficient": lasso_coefs,
                                "irr": np.exp(lasso_coefs), "selected": np.abs(lasso_coefs) > 1e-12,
                                "unit": [metadata[name]["unit"] for name in lasso.feature_names]})
    gain = booster.get_score(importance_type="gain")
    gains = pd.DataFrame({"feature": names, "gain": [float(gain.get(name, 0.)) for name in names]})
    gains["gain_share"] = gains["gain"] / gains["gain"].sum() if gains["gain"].sum() > 0 else 0.
    gains = gains.sort_values("gain", ascending=False).reset_index(drop=True)
    p = len(beta)
    summary = {
        "n_rows": len(frame), "n_features": len(names), "poisson_n_parameters": p,
        "lasso_alpha": alpha, "lasso_nonzero_coefficients": int(lasso_table["selected"].sum()),
        "xgb_rounds": rounds, "poisson_dropped_aliases": poisson.dropped_aliases,
        "poisson_fit_warnings": poisson.fit_warnings, "lasso_fit_warnings": lasso.fit_warnings,
        "poisson_pearson_dispersion": float(np.sum((y - mu) ** 2 / mu) / (len(frame) - p)),
        "hyperparameter_rule": "Median inner-selected hyperparameters; geometric median for alpha.",
        "role": "Full-data conditional associations and fitted artifacts, not held-out performance.",
        "main_offset": "zero; log1p_views is estimated as a predictor",
        "numeric_coefficient_scale": "Original feature units shown in coefficient tables; glum reverses internal scaling.",
        **covariance_info,
    }
    bundle = ModelBundle(builder, poisson, lasso, booster, asdict(cfg))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        coefficients.to_csv(directory / "poisson_coefficients.csv", index=False)
        lasso_table.to_csv(directory / "lasso_coefficients.csv", index=False)
        gains.to_csv(directory / "xgb_feature_importance.csv", index=False)
        _save_json(directory / "final_fit_summary.json", summary)
        _save_json(directory / "feature_reference_levels.json", builder.category_maps_)
        _save_json(directory / "svd_top_terms.json", builder.topic_terms())
        _save_json(directory / "final_feature_audit.json", builder.diagnostics())
        joblib.dump(bundle, directory / "model_bundle.joblib", compress=3)
    return {"summary": summary, "coefficients": coefficients, "lasso_coefficients": lasso_table,
            "xgb_importance": gains, "bundle": bundle}
