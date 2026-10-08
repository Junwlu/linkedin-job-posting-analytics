# Validation record

Validation date: **October 8, 2026**. This record applies to the first public reconstruction, not the historical course submission.

| Check | Evidence |
| --- | --- |
| Full source-to-results run | Completed on Windows, Python 3.11.5; 624.05 seconds; source fingerprint matches the recovered canonical file |
| Input integrity | 123,849 unique source jobs; 23,319 eligible rows; exact CSV SHA-256 in `results/data_audit.json` |
| Held-out results | Five saved folds, nine models/specifications, complete primary-model coverage |
| Independent metric checks | Arithmetic recomputation of pooled and fold-level MAE/RMSE/Poisson deviance; all passed; OOF hash stored in `results/independent_validation.json` |
| Unit and integration tests | 38 tests passed, including synthetic complete CV, model serialization, known-answer metrics, leakage boundaries, corrupt-output detection and downloader safeguards |
| Robust coefficient covariance | Independently compared with a separate statsmodels cluster-sandwich formula, including strongly differing feature scales |
| Model diagnostics | Final GLM fits report no convergence warnings; scaled Hessian condition 144.76; full design rank 73 |
| Notebook | Executed top to bottom: 12 code cells, no errors, six plots; calibration independently reconciled with saved chart data |
| Report | All displayed metric and coefficient rows checked against result CSVs; eight rendered PDF pages visually inspected |
| Packaging | Editable package installation, command-line help and dependency consistency check passed |
| Public-content review | Raw/processed text and model binaries excluded; no local user paths, credential material or original course documents included |

The notebook's six static figures were visually checked. A complete browser rendering of the notebook HTML was unavailable in the authoring environment; execution, notebook-format checks and embedded figure inspection passed. The PDF was rendered with Poppler; no clipped or overlapping content was found.

The report builder rechecks current saved predictions, their validation fingerprint, the source hash, the configuration hash and population consistency before writing the report. It refuses a smoke run or an unverified source input.

## What these checks do not establish

They do not prove the missingness mechanism, historical event semantics, causal validity, representativeness, future-date performance or company-disjoint generalization. They also do not prove bit-identical floating-point results on every hardware/library combination. The original R execution was not exactly reconstructed; the new pipeline corrects its specification and evaluation problems.

The GitHub workflow uses small fixtures and synthetic data. It does not download the full Kaggle file or rerun the expensive analysis. Its current status is available under the repository's Actions tab.
