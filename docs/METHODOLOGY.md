# Methodology

This project estimates **recorded application counts conditional on having a recorded application count and view count**. It compares three prediction methods on the same held-out postings and describes full-sample associations separately. It does not estimate a causal effect of changing a job advertisement, the probability that a view converts to an application, or outcomes for postings whose application counts are missing.

## Source and eligible population

The authoritative input is the original UTF-8 `postings.csv` inside `postings.csv.zip`. The loose, previously edited CSV has a changed schema and malformed records; it is not used as the source for this reconstruction. `data.py` preserves source identifiers, applies explicit domain checks, records exclusions, and exposes one row per unique `job_id` as `row_id`.

The common analysis population requires an observed, finite, nonnegative integer `applies` and an observed, finite, nonnegative `views`. Missing application counts are excluded, never replaced with zero. Missing view counts are excluded from every primary model, rather than giving only some estimators a different population. Counts in `data_audit.json` and `cv_run_summary.json` document the actual run.

The source audit found only positive values among recorded application counts and view counts. The upstream collection code discards some zero/false values. Thus, missing outcomes cannot be interpreted as confirmed zero applications, and this dataset does not support evaluation of zero-application detection. The Poisson objective is used as a working loss for conditional mean prediction in the observed sample. It is not a fitted model of the missingness mechanism or a validated full-population Poisson count distribution.

Remote-work labels distinguish recorded remote status from unknown status. Unknown remote status is not treated as evidence of on-site work. A constant sponsored field contributes no predictive information and is removed as a constant training feature. Non-USD salary values are excluded from the salary feature rather than mixed with USD values; their posting rows can remain eligible.

## What is modeled

The primary mean specification is

\[
\log\{E[Y_i\mid X_i,\text{recorded sample}]\}
=\beta_0 + f(X_i),
\]

where the predictors include `log1p_views = log(1 + views)`. For the two linear models, its coefficient is estimated rather than fixed at one. XGBoost can learn a nonlinear function of the same predictor. The primary offset is zero in all three models.

Views are counts recorded alongside the applications; their exact historical counting window and relisting behavior are unverified. They are not known at the time a new posting is created. Consequently, this is retrospective prediction conditional on recorded reach, not a forecast using only information available before publication. View counts may also reflect job attractiveness and other processes related to applications, so controlling for them does not identify a causal content effect.

### Why the old time exposure was changed

The historical analysis used

\[
\operatorname{offset}_i=\log\{\mathrm{days}_i(1+\mathrm{views}_i)\}.
\]

This fixes the coefficients of both log duration and log cumulative views to one. Because cumulative views already reflect accumulated reach, multiplying them by duration imposes an additional time factor; under a roughly constant viewing rate it can count duration twice. This is a substantive assumption, not an automatically justified exposure measure.

More importantly, `expiry` is a planned endpoint and does not establish when application/view counts were observed. The data contain no verified common collection timestamp that would justify substituting today's date or the file download date. The reconstruction therefore does not call an expiry interval “observed days open.”

Two predeclared Poisson sensitivity analyses retain the older assumptions explicitly:

| Analysis | Ordinary view predictor | Offset | Eligible population |
|---|---|---|---|
| Primary learned-views model | `log(1 + views)` | 0 | Common observed-target/observed-views rows |
| Views-offset sensitivity | Omitted | `log(1 + views)` | Same common rows |
| Legacy interval sensitivity | Omitted | `log(legacy_days_open) + log(1 + views)` | Rows with valid positive planned endpoint intervals |

The legacy analysis uses the earliest valid closed/expiry endpoint relative to listing time, subject to the explicit validity rule in `data.py`. It does not impute a missing endpoint with the current date. For a fair comparison, a separate learned-views Poisson model is fitted and tested on exactly the same legacy-eligible training and test rows. The main-model score on a larger population is not used as the sole comparator for this sensitivity.

## Predictors and training-only transformations

`FeatureBuilder` does not use `applies`, any transformed target, or job/company identifiers as predictors. It preserves row count and order, validates finite output, and cannot silently delete rows during design-matrix construction.

Pure fixed rules, such as domain validation and title-to-role mappings, may be applied before splitting. Estimated quantities are learned only on the relevant training rows: salary cutoffs, retained category levels, TF-IDF vocabulary/IDF, SVD components, variable-column checks, and estimator scaling.

| Predictor block | Construction | Coefficient unit |
|---|---|---|
| View count | `log(1 + views)`, primary models only | One log unit; equivalently, multiplying `1 + views` by approximately 2.718 |
| Title length | Number of title characters divided by 10 | 10 title characters |
| Description length | Number of description characters divided by 1,000 | 1,000 description characters |
| Title keywords | Fixed boundary-aware indicators for data, analyst, engineer, developer, manager, nurse, sales, marketing, Python, SQL, Java, cloud, AWS, Azure and GCP | Indicator changing from 0 to 1 |
| Salary band | Training-sample 33rd and 67th percentiles of valid salary, plus `Unknown` | Named level versus recorded reference |
| Categories | Application type, work type, experience level, remote status, sponsored status, region, title-derived seniority and role family | Named level versus recorded reference |
| Description topics | Training-only TF-IDF with English stop words, unigrams/bigrams and the configured vocabulary cap, followed by up to 15 truncated SVD components | One component-score unit |

The vocabulary cap, minimum document frequency, rare-category minimum and SVD count are recorded in `config.json`. Small synthetic tests may use lower settings. SVD components are statistical directions, not manually validated semantic topics; their signs and loadings should not be interpreted as inherent positive/negative sentiment. `svd_top_terms.json` records signed loadings for inspection.

`skill_hits` is omitted because it is the sum of the included keyword indicators. `is_intern` is omitted because it repeats title-derived seniority information. Constant columns are removed using training rows. For unpenalized Poisson fits, additional exact linear dependencies are removed by a rank-revealing QR decomposition of the centered/scaled training design; omitted aliases are recorded.

### Categories and references

The preferred references are OffsiteApply, Full-time, Mid-Senior level, Unknown remote status, NotSponsored, South, Unknown seniority, Other role family, and Med salary band. These are preferences, not assertions that every level exists in every training sample. If a preferred reference is absent, an observed training reference is selected and recorded. No synthetic on-site group is invented.

Rare levels are grouped using training counts. A held-out level not retained during training maps to the training `__Other__` group if that group exists, otherwise to the recorded training reference. The number of such fallbacks is reported in the feature audit. This rule preserves rows and makes the behavior for unseen levels explicit; it does not estimate an independent effect for an unseen level.

`feature_reference_levels.json` records the final fit's actual levels and references. Fold-specific mappings are stored in `feature_audit.json`.

## Estimators and comparators

### Unpenalized Poisson GLM

The log-linear model uses the count target and a log link, without an L1/L2 penalty (`glum` `alpha=0`). The primary model learns the views coefficient. The sensitivity models receive an additive log exposure offset at both fitting and prediction.

### Lasso-Poisson

`glum.GeneralizedLinearRegressor` is fitted with `family="poisson"`, `link="log"`, and `l1_ratio=1`. This is an L1-penalized Poisson GLM, not an L2 Poisson estimator relabeled as Lasso. Its objective is

\[
\frac{D(y,\mu)}{2n}+\lambda\|\beta\|_1.
\]

In `glum`, `alpha` is the penalty strength corresponding to glmnet's lambda. The intercept is unpenalized. Predictor scaling is learned internally on the training matrix so penalty comparisons do not depend on arbitrary feature units. Reported coefficients are returned in the original feature units listed above. The default candidate penalties are 0.001, 0.01, 0.1 and 1.0. Selected Lasso coefficients are reported without naive post-selection significance intervals. The [official glum API](https://glum.readthedocs.io/en/latest/glm.html) documents the objective, offset arguments and coefficient scaling.

### XGBoost-Poisson

The boosted-tree model uses `count:poisson`, `tree_method="hist"`, and the same primary feature information as the linear models. Tree depth, learning rate, sampling fractions and thread limits are fixed in the implementation/configuration. Only the number of boosting rounds is selected on inner validation, with the configured maximum and early-stopping patience.

The DMatrix receives the corresponding base margin both during training and prediction. It is zero for the primary learned-views model. A Poisson log exposure offset would be passed as the base margin, not as a response-scale exposure, and predicted counts must not be multiplied by exposure a second time. This follows [XGBoost's offset documentation](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html).

The final importance table uses **average split gain** from the final fitted booster, normalized across features. It is not a total-gain contribution decomposition, a percentage of explained outcome variation, or evidence of a causal effect.

### Baselines and text ablation

The constant-mean baseline predicts the outer training mean for every held-out row. The constant-exposure baseline estimates `sum(applies) / sum(1 + views)` on the outer training rows and multiplies that rate by each held-out row's `1 + views`.

The `poisson_no_svd` ablation keeps the same structured variables, title keyword indicators, text lengths and learned view predictor, but omits description SVD components. Its paired comparison with the full Poisson model asks whether those components add predictive information. It is **not** a comparison against a model with all text-derived information removed: titles also supply keywords, seniority and role family.

## Validation and tuning

1. Generate one shuffled five-fold assignment with seed `20251201`, save it by `row_id`, and reuse it for every model. These are posting-level splits, not company-disjoint or chronological splits.
2. In each outer training fold, create one reproducible 20% inner validation holdout. Fit a separate feature builder on the inner training rows only, then transform inner validation.
3. Select the Lasso penalty by inner validation mean Poisson deviance. Exact ties choose the larger penalty. Select XGBoost rounds by inner validation Poisson negative log likelihood. On the same target rows, this differs from deviance only by target-dependent constants and scale.
4. Discard the inner fitted estimators. Fit a fresh feature builder on all outer training rows, refit each selected model, and predict the untouched outer test rows. Outer tests never choose penalty strengths or stopping rounds.
5. Pool the resulting out-of-fold predictions so that every eligible primary row has exactly one prediction from each primary estimator.

This is an outer five-fold evaluation with an inner holdout for tuning, rather than a full inner five-fold search. The fixed design keeps tuning separate from reported test performance while remaining practical on a local computer. The procedure follows the [training-only preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).

`cv_oof_predictions.csv` is a wide table with `row_id`, observed counts/views, fold, and one `pred_*` column per comparator. `fold_assignments.csv` preserves the split. Legacy-ineligible rows have missing legacy predictions, never zero-filled predictions.

## Metrics

Every model is evaluated on raw application-count predictions using the same definitions:

\[
\operatorname{MAE}=\frac1n\sum_i|y_i-\hat\mu_i|,
\qquad
\operatorname{RMSE}=\sqrt{\frac1n\sum_i(y_i-\hat\mu_i)^2},
\]

\[
\operatorname{MeanPoissonDeviance}
=\frac2n\sum_i\left[y_i\log\left(\frac{y_i}{\hat\mu_i}\right)-y_i+\hat\mu_i\right].
\]

For `y=0`, the logarithmic term is defined as zero. Predicted means must be finite and strictly positive; invalid predictions cause an error rather than being silently clipped or excluded. Observed targets are not winsorized. The mean deviance is the primary common loss for tuning and comparison; MAE and RMSE show different sensitivity to errors and large counts.

`cv_summary.csv` contains **pooled OOF metrics**. It does not average fold RMSE and call the result pooled RMSE, and it does not compare a sum for one model with a mean for another. A separately labeled deviance sum is saved for reconciliation. `cv_fold_metrics.csv` preserves each fold's metrics, including the number of evaluated rows.

The historical results used different preprocessing, target populations, offset assumptions and deviance aggregation. A percentage improvement over the historical table is therefore not a controlled estimate of the benefit of this corrected pipeline. Comparisons within the newly saved shared-fold experiment are the appropriate evidence.

## Full-data association estimates and uncertainty

After OOF evaluation, `fit_interpretation` fits new models on the full eligible dataset. Final penalty strength is the geometric median of the inner-selected penalties; the number of boosting rounds is the rounded median of the inner-selected round counts. Outer test errors do not choose these full-data settings.

The final unpenalized Poisson coefficient table exponentiates coefficients to show conditional mean ratios, often called IRRs in Poisson models. Each row lists its feature unit. These are sample-conditional associations, not causal effects. An interval spanning one does not establish a directional association at the reported 95% level.

Standard errors use a sandwich covariance clustered by recorded company identifier. Missing company identifiers are treated as separate unknown clusters, not one artificial company. If usable multiple clusters are unavailable, the implementation uses HC1 robust covariance and records that choice. The finite-sample cluster correction is

\[
\frac{M}{M-1}\frac{n-1}{n-p},
\]

where `M` is the number of clusters and `p` includes the intercept. The inference calculation centers/scales the design for numerical stability, solves the full-rank Hessian, then maps covariance back to the original feature units. Nonfinite, excessively ill-conditioned or non-positive-semidefinite covariance estimates are rejected instead of producing unsupported intervals. The implementation is checked against an independent `statsmodels` cluster-covariance calculation.

These intervals are descriptive: they do not account for target-selection bias, uncertainty in the source measurement process, or data-driven inspection of many coefficients. They also treat the learned feature representation as fixed. The full-data Pearson dispersion statistic is a diagnostic of residual variation, not a proof that an exact Poisson distribution is appropriate for the selected positive-count sample.

Final fitted associations and feature importance are saved separately from OOF results. `model_bundle.joblib` includes fitted preprocessing and all three primary estimators. It is intended for trusted local use; loading a pickle/joblib artifact from an unknown source is unsafe.

## Reproducibility checks

The test suite covers training-only vocabulary and salary cutoffs, explicit reference categories, unseen-category row preservation, missing-count/view exclusions, finite Poisson predictions, exact zero handling in deviance, the offset doubling property, Lasso sparsity under a large penalty, deterministic shared folds, inner/outer row separation, paired legacy eligibility, company-cluster covariance against an independent library, and prediction equality after model serialization/reload.

The independent result verifier recomputes reported metrics directly from saved OOF rows. Package versions, configuration, source audit and run artifacts should accompany any reported numbers. A completed source file or a synthetic smoke test alone is not evidence that a full-data result was run successfully.
