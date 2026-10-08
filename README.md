# LinkedIn Job Posting Analytics

A reproducible, source-audited reanalysis of **recorded application counts** in a Kaggle LinkedIn dataset. This reconstructs a December 2025 course project from a recovered original file and corrects its data, exposure, evaluation and interpretation problems. The reconstruction was completed in October 2026.

[中文说明](README.zh-CN.md) · [Corrected report (PDF)](reports/corrected_report.pdf) · [Report source](reports/corrected_report.md) · [Executed notebook](notebooks/reanalysis.ipynb) · [Corrections](docs/CORRECTIONS.md)

## What this project establishes

- **123,849 unique raw postings; 23,319 eligible observations.** The target and views are both recorded for only 18.8% of the source. Every observed target is positive; missing counts are not assumed to be zero.
- **Three comparable count-mean models:** Poisson GLM, L1-penalized Poisson and XGBoost. Learned transformations and tuning stay within training splits, and all primary models use the same saved outer folds.
- **A corrected visibility specification:** learn the coefficient of `log1p(views)` rather than assuming an unverified `days_open * (views + 1)` exposure denominator. Fixed-offset versions are explicit sensitivity checks.
- **An inspectable evidence chain:** source fingerprint, cleaning audit, row-level out-of-fold predictions, independently recomputed metrics, descriptive coefficient intervals, figures, report and notebook.

This is retrospective conditional prediction: views is measured alongside applications. It is not a pre-posting forecast, a causal optimization study or a model of completed applications across all application flows.

## Results

The generated [model table](results/cv_summary.csv) reports pooled MAE, RMSE and **mean** Poisson deviance on identical held-out observations. See the report for the executed values, ablation and matched-population sensitivity comparisons.

| Model | MAE | RMSE | Mean Poisson deviance |
| --- | ---: | ---: | ---: |
| Poisson GLM | 4.987 | 23.004 | 4.018 |
| L1 Poisson | 4.974 | 22.906 | 3.993 |
| XGBoost | **4.106** | **15.530** | **3.026** |

XGBoost MAE was **17.7% lower than Poisson GLM**, and 66.6% lower than the training-mean baseline. Description SVD reduced Poisson MAE by 2.40% in the paired ablation. These are results on the selected observed-count population, not changes in real applicant volume or comparisons with the damaged historical run.

![Shared-fold model comparison](results/figures/model_comparison.png)

## Reproduce

Python **3.11** was used for the recorded run. The full analysis requires roughly 1 GB for raw/uncompressed input and additional working space. The recorded four-thread CPU run took **10.4 minutes** including five outer folds, inner tuning, final fits and figures; runtime varies by hardware. No GPU or paid service is required.

```bash
git clone https://github.com/Junwlu/linkedin-job-posting-analytics.git
cd linkedin-job-posting-analytics
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -r requirements.lock.txt
python -m pip install -e ".[download,notebook,report,test]"
python scripts/download_data.py
python -m linkedin_analytics --data data/raw/postings.csv --config config.json --output results
python scripts/build_report.py
python scripts/build_notebook.py --execute
python -m pytest -q
```

`requirements.lock.txt` records the analysis and supporting tool environment, including downloader/report packages. Wheels, BLAS libraries and small floating-point differences can vary by platform. The report independently rechecks saved predictions and does not refit models. For a fresh compatible environment rather than the recorded versions, install the project extras without the lock file. CI checks synthetic fixtures on Linux; the full recorded analysis ran on Windows.

The downloader requests **`arshkon/linkedin-job-postings/versions/13`**, verifies the original file's exact byte count and SHA-256, and refuses to overwrite an existing mismatched file. Kaggle may require account acceptance/authentication; follow its normal local setup and never commit credentials. If you already have the matching CSV or original ZIP:

```bash
python -m linkedin_analytics --data "/path/to/postings.csv.zip" --config config.json --output results
```

The fingerprint is:

```text
de33f27c8d1b226014e0cede003404e9b0acb66db718671fda9b3e3538eca6ea
```

The local raw member matches Kaggle v13's published file size; the inspected manifest did not expose a remote cryptographic hash. A fingerprint mismatch is a source change to investigate, not an error to suppress. See [source details](docs/DATA_SOURCE.md) and [acquisition instructions](data/README.md).

For a pipeline smoke run, keep the output separate from full results:

```bash
python -m linkedin_analytics --data data/raw/postings.csv --smoke-rows 800 --output results/smoke
```

This still validates the complete input before taking a fixed sample. Smoke outputs must not be used as the reported result.

## Read the project

| Path | Purpose |
| --- | --- |
| `src/linkedin_analytics/data.py` | ZIP/CSV parsing, input validation, deterministic cleaning and audit |
| `src/linkedin_analytics/features.py` | Training-only salary/category transforms, title rules, TF-IDF and SVD |
| `src/linkedin_analytics/modeling.py` | Shared CV, inner selection, models, sensitivity fits and robust intervals |
| `src/linkedin_analytics/validation.py` | Independent metric arithmetic and saved-result integrity checks |
| `src/linkedin_analytics/plotting.py` | Figures and chart source tables from result files |
| `config.json` | Seed, folds, hyperparameter grid, feature capacity and thread settings |
| `results/` | Audits, OOF predictions, coefficients, reference levels and figures |
| `notebooks/reanalysis.ipynb` | Executed, editable results review; does not silently repeat expensive CV |
| `scripts/build_report.py` | Regenerates Markdown and PDF from the actual result tables |
| `tests/` | Small fixtures and synthetic model checks; no Kaggle download needed |
| `docs/METHODOLOGY.md` | Exact model contract, transformations, units and comparison rules |

The raw CSV, processed job descriptions and local serialized model are ignored by Git. Outputs include public source job IDs for split auditing, not full descriptions. The original R and Word/PPT working files are archived separately; this repository's maintained pipeline is a Python reimplementation, not a claim that the old R execution was reproduced exactly. No SQL workflow is claimed.

## Interpretation boundaries

1. Outcome missingness differs sharply by application flow. Results concern the selected recorded-outcome cohort.
2. The source collector may have dropped zeros and Boolean false values. Missing remote flags are `Unknown`, not verified onsite work.
3. Offsite application events can reflect apply clicks rather than completed applications. The historical event semantics were not verified record by record.
4. The CSV omits a job capture timestamp. Expiry is not observed follow-up duration.
5. Random posting folds permit companies and similar text to appear in training and test. Future-time and company-disjoint performance remain untested.
6. Full-sample coefficient intervals are nominal company-cluster robust intervals. They are descriptive, not causal or adjusted for all feature screening. XGBoost importance is normalized average split gain, not a causal contribution.

## Attribution and license

Dataset: Arsh Koneru, [LinkedIn Job Postings (2023-2024), Kaggle version 13](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings); the data card also credits `@zoeyyuzou`. Source data and distributed derived analytical outputs are attributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). New analysis code is [MIT licensed](LICENSE). See [the license boundary](docs/DATA_LICENSE.md) and [citation metadata](CITATION.cff).
