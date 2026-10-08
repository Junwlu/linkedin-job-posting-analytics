"""Figures derived solely from saved, validated analytical outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"constant_mean": "Training mean", "constant_exposure": "Proportional (views + 1)",
          "poisson": "Poisson GLM", "lasso_poisson": "L1 Poisson", "xgboost": "XGBoost",
          "poisson_no_svd": "Poisson without SVD"}
COLORS = ["#3566a6", "#159085", "#df8850"]


def build_figures(output_dir):
    root = Path(output_dir)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    audit = json.loads((root / "data_audit.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(root / "cv_oof_predictions.csv")
    summary = pd.read_csv(root / "cv_summary.csv").set_index("model")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 120})

    def save(fig, name):
        fig.savefig(figures / name, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    observations = pd.DataFrame(audit["applies_observability_by_application_type"]).T
    observations.to_csv(root / "outcome_observability.csv", index_label="application_type")
    fig, ax = plt.subplots(figsize=(8.4, 3.6), layout="constrained")
    bars = ax.barh(observations.index, 100 * observations.observed_fraction, color="#3566a6")
    ax.bar_label(bars, labels=[f"{100*r.observed_fraction:.1f}% ({int(r.observed_applies):,}/{int(r.raw_rows):,})"
                              for r in observations.itertuples()], padding=5, fontsize=9)
    ax.set(xlim=(0, 105), xlabel="Rows with a recorded application count (%)",
           title="Outcome availability differs by application flow")
    ax.grid(axis="x", alpha=.15)
    save(fig, "outcome_observability.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].hist(np.log10(oof.applies), bins=35, color="#3566a6", alpha=.9)
    axes[0].set(xlabel="log10(recorded applications)", ylabel="Postings",
                title="Only positive outcomes are observed")
    hb = axes[1].hexbin(np.log10(oof.views + 1), np.log10(oof.applies),
                        gridsize=45, bins="log", mincnt=1, cmap="viridis")
    axes[1].set(xlabel="log10(views + 1)", ylabel="log10(recorded applications)",
                title="Visibility is strongly related to counts")
    fig.colorbar(hb, ax=axes[1], label="Postings (log color scale)")
    save(fig, "count_distribution.png")

    models = [m for m in ("constant_mean", "poisson", "lasso_poisson", "xgboost", "poisson_no_svd") if m in summary.index]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    for ax, metric, title in zip(axes, ["mae", "rmse", "mean_poisson_deviance"],
                                 ["MAE", "RMSE", "Mean Poisson deviance"]):
        bars = ax.barh([LABELS[m] for m in models], summary.loc[models, metric], color="#3566a6")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, float(summary.loc[models, metric].max()) * 1.22)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    fig.suptitle(f"Shared five-fold predictions | {len(oof):,} postings | lower is better", fontsize=13)
    save(fig, "model_comparison.png")

    calibration = []
    fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained")
    for model, color in zip(["poisson", "lasso_poisson", "xgboost"], COLORS):
        pred = oof["pred_" + model]
        group = pd.qcut(pred.rank(method="first"), 10, labels=False)
        table = pd.DataFrame({"predicted": pred, "observed": oof.applies, "decile": group}).groupby("decile").agg(
            predicted_mean=("predicted", "mean"), observed_mean=("observed", "mean"), n=("observed", "size")).reset_index()
        table["model"] = model
        calibration.append(table)
        ax.plot(table.predicted_mean, table.observed_mean, "o-", color=color, label=LABELS[model])
    cal = pd.concat(calibration, ignore_index=True)
    cal.to_csv(root / "calibration_deciles.csv", index=False)
    lim = max(cal.predicted_mean.max(), cal.observed_mean.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#8c98a5", label="Equal predicted / observed")
    ax.set(xlabel="Mean out-of-fold prediction", ylabel="Mean recorded applications",
           title="Calibration by predicted-count decile", xlim=(0, lim), ylim=(0, lim))
    ax.legend(fontsize=8)
    save(fig, "calibration.png")

    coef_path = root / "poisson_coefficients.csv"
    if coef_path.exists():
        coefficients = pd.read_csv(coef_path)
        # Pre-specified interpretable fields, not selection by p-value.
        selected = coefficients[coefficients.feature.str.contains(
            "application_type|formatted_work_type|kw_python|kw_sql|log1p_views", regex=True)].copy()
        selected = selected[~selected.feature.str.contains("__Other__", regex=False)]
        selected = selected[np.isfinite(selected.ci_lower) & np.isfinite(selected.ci_upper) & selected.ci_lower.gt(0)]
        if len(selected):
            fig, ax = plt.subplots(figsize=(9.5, max(3.5, len(selected) * .36)), layout="constrained")
            ax.errorbar(selected.irr, np.arange(len(selected)),
                        xerr=np.vstack([selected.irr-selected.ci_lower, selected.ci_upper-selected.irr]),
                        fmt="o", capsize=3, color="#159085")
            ax.set_yticks(np.arange(len(selected)), selected.feature)
            ax.axvline(1, color="#9ca5af", linestyle="--")
            ax.set_xscale("log")
            from matplotlib.ticker import NullFormatter, FixedLocator, ScalarFormatter
            ax.xaxis.set_major_locator(FixedLocator([.5, 1, 1.5, 2, 3]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set(xlabel="Conditional count ratio (95% company-cluster robust CI)",
                   title="Selected full-sample Poisson associations")
            ax.invert_yaxis()
            save(fig, "selected_associations.png")
    importance_path = root / "xgb_feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).nlargest(12, "gain_share").sort_values("gain_share")
        fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
        ax.barh(importance.feature, 100 * importance.gain_share, color="#df8850")
        ax.set(xlabel="Share of normalized average split gain (%)",
               title="Full-sample XGBoost: model reliance, not causal effects")
        save(fig, "feature_importance.png")
    return sorted(str(p.relative_to(root)) for p in figures.glob("*.png"))
