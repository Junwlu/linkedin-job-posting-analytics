"""Build the corrected Markdown and PDF reports from the executed result files.

Run from any working directory: python scripts/build_report.py
Requires the project's [report] extra. Numerical claims are read from results/.
"""
from __future__ import annotations

import html
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                               TableStyle, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from linkedin_analytics.validation import verify_results
NAVY, TEAL = colors.HexColor("#193651"), colors.HexColor("#167a78")


def build():
    REPORTS.mkdir(exist_ok=True)
    audit = json.loads((RESULTS / "data_audit.json").read_text(encoding="utf-8"))
    fit = json.loads((RESULTS / "final_fit_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((RESULTS / "independent_validation.json").read_text(encoding="utf-8"))
    current_validation = verify_results(RESULTS)
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("oof_sha256") != current_validation["oof_sha256"]:
        raise ValueError("Saved independent validation is stale; rerun the pipeline.")
    if metadata.get("smoke_sample") or not audit["matches_original_v13_fingerprint"]:
        raise ValueError("This report requires the full pinned source run, not a smoke/new-source run.")
    if metadata["source_csv_sha256"] != audit["raw_dataset_sha256"]:
        raise ValueError("Source fingerprint differs between audit and run metadata.")
    if metadata["config_sha256"] != hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest():
        raise ValueError("Configuration has changed since the completed run.")
    summary = pd.read_csv(RESULTS / "cv_summary.csv").set_index("model")
    coef = pd.read_csv(RESULTS / "poisson_coefficients.csv").set_index("feature")
    oof = pd.read_csv(RESULTS / "cv_oof_predictions.csv")
    tuning = pd.read_csv(RESULTS / "cv_tuning.csv")
    n = len(oof)
    if fit["n_rows"] != n or n != audit["main_model_rows"]:
        raise ValueError("Full-fit, held-out and data-audit populations differ.")
    rows, story, md = audit["raw_rows"], [], []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=27,
                              leading=31, textColor=NAVY, spaceAfter=16))
    styles.add(ParagraphStyle("SectionTitle", fontName="Helvetica-Bold", fontSize=18,
                              leading=22, textColor=NAVY, spaceAfter=12))
    styles.add(ParagraphStyle("BodyReport", fontName="Helvetica", fontSize=10.1,
                              leading=14.6, spaceAfter=9, textColor=colors.HexColor("#253442")))
    styles.add(ParagraphStyle("CaptionReport", fontName="Helvetica", fontSize=8.4,
                              leading=11.2, spaceAfter=8, textColor=colors.HexColor("#536679")))
    styles.add(ParagraphStyle("CellReport", fontName="Helvetica", fontSize=8.3, leading=10.5))
    styles.add(ParagraphStyle("CodeReport", fontName="Courier", fontSize=8.2, leading=11,
                              backColor=colors.HexColor("#edf3f7"), borderPadding=8, spaceAfter=10))

    def para(text, style="BodyReport"):
        md.append(text + "\n")
        story.append(Paragraph(html.escape(text), styles[style]))

    def section(title, first=False):
        if not first:
            story.append(PageBreak())
        md.append("\n## " + title + "\n")
        story.append(Paragraph(html.escape(title), styles["SectionTitle"]))

    def table(headers, data, widths=None):
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in data:
            md.append("| " + " | ".join(str(x) for x in row) + " |")
        md.append("")
        cells = [[Paragraph(html.escape(str(c)), styles["CellReport"]) for c in r]
                 for r in [headers] + data]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3eef4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), .8, TEAL),
            ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#d8e2e8")),
        ]))
        story.extend([t, Spacer(1, 11)])

    def figure(filename, caption, max_height=265):
        path = RESULTS / "figures" / filename
        w, h = ImageReader(str(path)).getSize()
        scale = min(490 / w, max_height / h)
        md.append(f"![{caption}](../results/figures/{filename})\n")
        story.append(KeepTogether([Image(str(path), width=w*scale, height=h*scale),
                                  Paragraph(html.escape(caption), styles["CaptionReport"])]))

    def value(model, metric):
        return float(summary.loc[model, metric])

    main_names = {"constant_mean": "Training-mean baseline", "constant_exposure": "Proportional (views + 1) baseline",
                  "poisson": "Poisson GLM", "lasso_poisson": "L1 Poisson", "xgboost": "XGBoost",
                  "poisson_no_svd": "Poisson without description SVD"}
    mae_reduction = 100 * (1-value("xgboost", "mae")/value("constant_mean", "mae"))
    svd_delta = 100 * (1-value("poisson", "mae")/value("poisson_no_svd", "mae"))
    best = {metric: min(("poisson", "lasso_poisson", "xgboost"), key=lambda m: value(m, metric))
            for metric in ["mae", "rmse", "mean_poisson_deviance"]}

    md.append("# Recorded LinkedIn Job Applications\n")
    md.append("Source-audited reanalysis of a December 2025 course project. Revision: October 8, 2026.\n")
    story.append(Paragraph("Recorded LinkedIn<br/>Job Applications", styles["ReportTitle"]))
    para("SOURCE-AUDITED REANALYSIS | OCTOBER 8, 2026", "CaptionReport")
    para("Junwei Lu | Reconstruction of the December 2025 course project", "CaptionReport")
    para("This report replaces unsupported numerical comparisons and exposure interpretations in the earlier report and presentation. It reconstructs the analysis from a recovered original input; it is not an exact replay of the historical execution state.")
    section("Executive findings", first=True)
    para(f"The recovered Kaggle file contains {rows:,} unique job postings. Only {n:,} ({100*n/rows:.1f}%) have both recorded applications and views. Every observed application count is positive. The analysis therefore describes a selected subset, with substantial uncertainty about missing outcomes.")
    ranking = (f"{main_names[best['mae']]} had the lowest MAE, RMSE and mean Poisson deviance."
               if len(set(best.values())) == 1 else
               f"{main_names[best['mae']]} had the lowest MAE, {main_names[best['rmse']]} the lowest RMSE, and {main_names[best['mean_poisson_deviance']]} the lowest mean Poisson deviance.")
    para(f"Three count-mean models were evaluated on the same five held-out folds. {ranking} XGBoost MAE was {value('xgboost','mae'):.3f}, a {mae_reduction:.1f}% reduction against the training-mean baseline on this selected sample.")
    para(f"The description SVD comparison changed Poisson MAE by {svd_delta:.2f}% relative to the same feature set without SVD (positive means improvement). Title keywords and text length remain in both models. The experiment isolates this particular description representation, not the value of all text information.")
    para("Views is a concurrent measurement and enters the primary model with a learned coefficient. Predictions describe recorded counts conditional on observed visibility; they are not forecasts available before a job is posted. Expiry dates do not provide a verified observation duration.")
    table(["Evidence", "Scope"], [
        [f"{rows:,} raw rows / {n:,} analyzed", "Outcome and views jointly observed"],
        ["Five shared outer folds", "Held-out postings within this collection"],
        ["Full-sample coefficient intervals", "Company-cluster robust, descriptive associations"],
        ["Reproducible code and saved predictions", "Pinned input fingerprint, fold-level preprocessing, independent metric checks"],
    ], [190, 300])
    para("The results do not identify causal effects, completed applications across every flow, zero-application behavior, or performance on future postings and unseen employers.")

    section("1. Data provenance and analysis population")
    para("Input: Arsh Koneru's LinkedIn Job Postings (2023-2024), Kaggle version 13, released August 19, 2024. The recovered archive member has 31 columns and 516,843,769 bytes, matching the version-13 file manifest. Its local SHA-256 is pinned in the downloader and data audit. A remote cryptographic hash was not supplied by the inspected manifest, so the version match rests on provenance and matching file size.")
    table(["Selection step", "Rows"], [
        ["Original rows / unique job IDs", f"{rows:,}"],
        ["Missing recorded application count", f"{audit['raw_applies_missing']:,}"],
        ["Observed valid application count", f"{audit['clean_observed_count_rows']:,}"],
        ["Excluded: missing/invalid views among those rows", str(audit['observed_applies_missing_or_invalid_views'])],
        ["Common modeling population", f"{n:,}"],
        ["Observed zero application counts", str(audit['observed_applies_zero'])],
    ], [365, 125])
    para("The earlier working CSV has 124,802 rows and 51 columns, including 33 unnamed headers, field displacement and duplicate records. It is not the verified original. The corrected pipeline starts from the intact archive, preserves job IDs, checks duplicate grain and validates observed counts.")
    figure("outcome_observability.png", "Figure 1. Availability of a recorded application count, not an application conversion rate.", 205)
    para("The collector code inspected for provenance drops false-like values, including zero. This could explain some missing counts, but the exact collector revision for every row is unknown. Missing outcomes are not globally imputed to zero. Missing remote flags remain Unknown rather than being called verified onsite jobs.")

    section("2. Measurement and statistical design")
    para("The response is the recorded `applies` field. LinkedIn's official descriptions distinguish completed applications from external apply clicks. Those definitions expose a measurement risk but do not prove the precise historical meaning of every record. Cross-flow coefficients can reflect measurement differences as well as applicant behavior.")
    para("For the Poisson GLM, log E[Y | X, observed Y and views] = intercept + beta_views * log(1 + views) + X beta. The main offset is zero. The Poisson loss is used as a working count-mean objective; the selected positive outcomes are not asserted to follow a standard Poisson distribution. Company-cluster robust intervals describe full-sample associations, conditional on the constructed features.")
    table(["Component", "Implementation"], [
        ["Predictors", "Application flow, work type, experience, remote flag, region, rule-based title categories, salary band, title/description lengths, title keywords, description SVD, log1p views."],
        ["Description features", "Training-only TF-IDF, at most 1,500 unigram/bigram terms; 15 SVD components. SVD axes are algebraic summaries, not validated semantic topics."],
        ["Learned transformations", "Vocabulary, SVD, salary quantiles, rare-category pooling and feature availability fitted within each training split. Fixed text rules use no outcome."],
        ["Outer evaluation", "Five shuffled posting-level folds, seed 20251201. Saved row IDs and identical test sets for all primary models."],
        ["Inner selection", "One 20% training-only holdout per outer fold. L1 alpha from 0.001, 0.01, 0.1, 1.0; XGBoost early stopping, maximum 800 rounds. Preprocessing is refitted on inner training."],
        ["Model fitting", "Unpenalized Poisson GLM, genuine L1 Poisson, and XGBoost count:poisson. All include learned log views; no held-out outcome tunes the outer fit."],
        ["Metrics", "Pooled out-of-fold MAE, RMSE and mean Poisson deviance. Deviance sum is also saved; all comparisons use a stated denominator."],
    ], [115, 375])
    para("Random posting folds can contain the same company and similar descriptions in training and test sets. They measure interpolation within this collection. Company-grouped and temporal validation were not performed, and group-robust coefficient intervals do not fix this prediction-validation limitation.")
    para("Primary ablation removes description SVD only. Sensitivity specifications fix the views coefficient at one, or additionally impose the earlier duration proxy. A companion learned-views model is evaluated on exactly the same duration-eligible subset.")

    section("3. Held-out model performance")
    table(["Model", "N", "MAE", "RMSE", "Mean deviance"], [
        [label, f"{int(summary.loc[model,'n']):,}", f"{value(model,'mae'):.3f}",
         f"{value(model,'rmse'):.3f}", f"{value(model,'mean_poisson_deviance'):.3f}"]
        for model,label in main_names.items()
    ], [220, 55, 66, 66, 83])
    figure("model_comparison.png", "Figure 2. Pooled predictions from the same held-out observations. Lower values are better.", 235)
    para(f"XGBoost reduces MAE by {mae_reduction:.1f}% relative to a constant training-mean prediction. This is a measured predictive comparison, not an improvement in real application volume. The best primary model by MAE is {main_names[best['mae']]}; metrics can rank models differently because large errors receive different weights.")
    para(f"With description SVD, Poisson MAE is {value('poisson','mae'):.3f}; without SVD it is {value('poisson_no_svd','mae'):.3f}. The corresponding mean deviances are {value('poisson','mean_poisson_deviance'):.3f} and {value('poisson_no_svd','mean_poisson_deviance'):.3f}. No formal uncertainty interval for the performance difference was estimated; this is a single, reproducible split design.")
    para(f"Final descriptive fits use the median of the inner-selected hyperparameters: L1 alpha {fit['lasso_alpha']:.3g}, XGBoost {fit['xgb_rounds']} rounds. Outer test losses are not used to select these final settings. Full-sample fits do not replace held-out scores.")

    section("4. Conditional associations and interpretation")
    selected = [name for name in ["application_type__ComplexOnsiteApply", "application_type__SimpleOnsiteApply",
                "formatted_work_type__Contract", "formatted_work_type__Part-time", "kw_python", "kw_sql", "kw_java", "log1p_views"] if name in coef.index]
    table(["Term", "Count ratio", "95% cluster CI"], [
        [name, f"{coef.loc[name,'irr']:.3f}", f"[{coef.loc[name,'ci_lower']:.3f}, {coef.loc[name,'ci_upper']:.3f}]"]
        for name in selected], [275, 85, 130])
    para("Application-flow references are OffsiteApply; work-type references are Full-time. Keyword indicators mean presence in the title, not verified applicant skills or all text mentions. All contrasts hold included features fixed. Full-time is a reference category, so a general claim that full-time roles outperform every alternative requires checking each contrast rather than reading a single coefficient.")
    para("Both recorded onsite-flow contrasts exceed one, but differences in event measurement and outcome availability prevent a causal interpretation. Contract and Part-time intervals include one, as do Internship and Temporary work types in the complete table. The earlier broad claim that full-time roles consistently attract stronger responses is therefore unsupported.")
    para("SQL and Java title indicators have positive conditional associations in this specification. Python's interval includes one. The earlier broad claim about explicit technical skill signals should be narrowed to the specific measured title indicators and qualified as an exploratory association.")
    if "log1p_views" in coef.index:
        beta = float(coef.loc['log1p_views','coefficient'])
        se = float(coef.loc['log1p_views','robust_std_error'])
        para(f"The fitted log1p-views coefficient is {beta:.3f} (95% cluster-robust interval {beta-1.96*se:.3f} to {beta+1.96*se:.3f}). It is an elasticity with respect to views + 1. The exponentiated coefficient instead describes multiplying views + 1 by e; it is not the effect of one additional view.")
    para(f"The full Poisson fit contains {fit['poisson_n_parameters']} parameters across {fit['n_clusters']:,} company/missing-ID clusters. Its Pearson dispersion statistic is {fit['poisson_pearson_dispersion']:.2f}, showing why naive Poisson standard errors should not be treated as adequate. The robust calculation is independently checked against a separate implementation.")
    para("These intervals are nominal and are not adjusted for screening many terms or for uncertainty in learned text features. They support cautious descriptive reporting, not confirmatory causal claims. The complete coefficient table and reference levels are supplied alongside the selected terms.")

    section("5. Sensitivity, calibration and model reliance")
    sensitivity = [("poisson", "Learned views, primary"), ("poisson_views_offset", "Fixed views coefficient = 1"),
                   ("poisson_legacy_matched_learned", "Learned views, duration-eligible"),
                   ("poisson_legacy_offset", "Legacy duration x views proxy")]
    table(["Specification", "N", "MAE", "RMSE", "Mean deviance"], [
        [label, f"{int(summary.loc[model,'n']):,}", f"{value(model,'mae'):.3f}",
         f"{value(model,'rmse'):.3f}", f"{value(model,'mean_poisson_deviance'):.3f}"]
        for model,label in sensitivity], [220,55,66,66,83])
    para("The legacy proxy uses the earlier of valid closure and expiry minus listing time, clipped to at least one day when the interval is positive. This deterministic interval can extend beyond the moment counts were captured. It is a stress test of an assumption, not recovered observed time at risk. Compare its result with the matched-subset learned-views row.")
    figure("calibration.png", "Figure 3. Means within prediction deciles, using out-of-fold predictions. The dashed line denotes equality.", 300)
    para("Decile calibration is a coarse diagnostic and may conceal poor predictions in tails or subgroups. Feature importance is supplied separately as normalized average split gain for the full-sample XGBoost fit. This describes model reliance, is sensitive to correlated predictors, and does not estimate intervention effects.")

    section("6. Corrections, limits and practical use")
    table(["Earlier issue", "Correction"], [
        ["Damaged working CSV", "Use the checksum-pinned original archive; retain job IDs and validate grain."],
        ["Unverified exposure adjustment", "Learn the log-views coefficient; show fixed exposure assumptions as sensitivities."],
        ["Text preprocessing before evaluation", "Fit text and category transformations separately inside every training split."],
        ["Inconsistent metric aggregation", "Save each OOF prediction and recompute pooled metrics independently with the same denominator."],
        ["Overstated feature claims", "State category references, robust uncertainty and measurement/selection limits. Avoid causal wording."],
        ["Mixed report/PPT result versions", "Generate this report from the same result files used by the notebook and figures."],
    ], [175,315])
    para("The collection has no documented probability sampling design. Missing outcomes may combine absent measurements and omitted zeros. Application-flow measurement may differ. Job-level capture times are unavailable. Salary values contain implausible extremes; training-derived bands reduce their numeric leverage but do not validate the salary source. All sponsored flags in this input are zero, so a sponsorship effect cannot be estimated.")
    para("A defensible use is retrospective reporting: audit missingness by flow, compare model errors on a common population and identify descriptive associations for further investigation. Recruiting teams would need verified completed-application tracking, capture timestamps, group/time validation and an intervention design before using these coefficients to prescribe posting changes.")
    para("The original course work included R, Python cleaning and notebooks. The reconstructed modeling pipeline is implemented in Python to provide a single executable environment. It retains the three model families but changes damaged inputs and invalid evaluation choices. SQL is not claimed as a demonstrated component because no supporting SQL workflow was recovered.")
    para("Next validation priorities are: recover observation timestamps and zero outcomes, reconcile event definitions by application flow, test future-date and company-disjoint splits, and evaluate robustness to alternative feature specifications. These are remaining research tasks, not completed validations.")

    section("7. Reproducibility and sources")
    para("Repository: https://github.com/Junwlu/linkedin-job-posting-analytics")
    para("Use Python 3.11. Create a virtual environment, install the pinned analysis environment and project extras, verify the source file, run the pipeline, then regenerate the report and notebook. Full commands and platform notes are in README.md. The raw data, processed descriptions and serialized model remain outside Git.")
    for command in ["python scripts/download_data.py", "python -m linkedin_analytics --data data/raw/postings.csv",
                    "python scripts/build_report.py", "python scripts/build_notebook.py --execute", "python -m pytest -q"]:
        md.append("```bash\n" + command + "\n```\n")
        story.append(Paragraph(html.escape(command), styles["CodeReport"]))
    para("Independent validation recomputes pooled and fold-wise losses directly from saved predictions, checks positive finite predictions, unique row coverage, split alignment and paired sensitivity populations. Synthetic tests cover cleaning, training-only transforms, genuine L1 sparsity, offset semantics, covariance calculations, downloader safeguards and detection of corrupted result files.")
    para("Primary sources are linked below. Source-specific details and exact provenance evidence are expanded in docs/DATA_SOURCE.md. Dataset citation: Koneru, Arsh (2024), LinkedIn Job Postings (2023-2024), version 13, Kaggle; data card additionally credits @zoeyyuzou. Data and derived analytical outputs retain CC BY-SA 4.0 attribution; newly written code is MIT licensed.")
    references = [
        ("Kaggle dataset and version history", "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings"),
        ("Collector schema and fixed source commit", "https://github.com/arshka/LinkedIn-Job-Scraper/tree/62c2e559234fe7b5cff3d902ccda9cea2d069c63"),
        ("LinkedIn applicant definitions", "https://www.linkedin.com/help/talent-insights/answer/a563146"),
        ("LinkedIn recruiter reporting definitions", "https://www.linkedin.com/help/linkedin/answer/a413441"),
        ("Dataset license", "https://creativecommons.org/licenses/by-sa/4.0/"),
        ("glum generalized linear model documentation", "https://glum.readthedocs.io/en/latest/glm.html"),
        ("XGBoost objective documentation", "https://xgboost.readthedocs.io/en/stable/parameter.html"),
    ]
    for label, url in references:
        md.append(f"- [{label}]({url})")
        story.append(Paragraph(f'<link href="{html.escape(url)}" color="#167a78">{html.escape(label)}</link>', styles["CaptionReport"]))
    md.append("")

    (REPORTS / "corrected_report.md").write_text("\n".join(md), encoding="utf-8")
    def footer(canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(colors.HexColor("#d9e3ea"))
        canvas.line(48, 43, width-48, 43)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5c6e7e"))
        canvas.drawString(48, 29, "LinkedIn recorded applications | Reanalysis 2026-10-08")
        canvas.drawRightString(width-48, 29, str(doc.page))
        canvas.restoreState()
    document = SimpleDocTemplate(str(REPORTS / "corrected_report.pdf"), pagesize=A4,
        rightMargin=48, leftMargin=48, topMargin=44, bottomMargin=60,
        title="Recorded LinkedIn Job Applications - Corrected Reanalysis",
        author="Junwei Lu")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Built reports/corrected_report.md and reports/corrected_report.pdf")


if __name__ == "__main__":
    build()
