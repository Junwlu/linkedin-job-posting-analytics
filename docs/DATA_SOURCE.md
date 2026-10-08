# Data source and measurement

This project reanalyzes the recovered original dataset. It does not reproduce the numerical results in the earlier presentation and report, which used a structurally damaged working CSV and inconsistent evaluation outputs. See [Corrections](CORRECTIONS.md).

## Source and version

| Item | Recorded value |
| --- | --- |
| Dataset | [LinkedIn Job Postings (2023 - 2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) |
| Publisher | Arsh Koneru, Kaggle account `arshkon` |
| Additional collector credited by the data card | `@zoeyyuzou` |
| Kaggle dataset ID | `3680745` |
| Pinned version | `13` |
| Release time | `2024-08-19T05:49:43.01Z` |
| Version note | `with zip code` |
| Input file | `postings.csv` |
| Expected file size | `516843769` bytes |
| Local raw-file SHA-256 | `de33f27c8d1b226014e0cede003404e9b0acb66db718671fda9b3e3538eca6ea` |
| Parsed raw population | `123849` records, `31` columns |
| Raw job identifiers | `123849` unique IDs, no duplicates |
| Data license | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

The [Kaggle metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/arshkon/linkedin-job-postings) and [version 13 file manifest](https://www.kaggle.com/api/v1/datasets/list/arshkon/linkedin-job-postings?datasetVersionNumber=13) were checked on October 8, 2026. The manifest's file size exactly matches the recovered archive member; version 12 instead reports `515197966` bytes. The hash above identifies the recovered local raw file. Kaggle's inspected manifest did not provide a remote content hash, so size matching is not a cryptographic proof of upstream identity.

The source covers listings from 2023 and 2024. Release dates are not job-level observation dates. Exact collection windows and a probability sampling frame were not established. The analysis does not represent all LinkedIn jobs or the entire labor market.

## What the recorded fields mean

The publisher's [field dictionary](https://github.com/arshka/LinkedIn-Job-Scraper/blob/62c2e559234fe7b5cff3d902ccda9cea2d069c63/DatabaseStructure.md) labels `applies` as an application count and `views` as a viewing count. `listed_time` and `original_listed_time` describe listing times, `expiry` describes expiration, and `closed_time` describes closure. Expiration is not the time at which the counts were observed.

The collector's [field mapping](https://github.com/arshka/LinkedIn-Job-Scraper/blob/62c2e559234fe7b5cff3d902ccda9cea2d069c63/json_paths/data_variables.csv) takes `data.applies` and `data.views` directly from LinkedIn's response. It does not convert views into views per day. Exact uniqueness, resetting after relisting, and the historical counting window remain unverified.

[LinkedIn's official applicant guidance](https://www.linkedin.com/help/talent-insights/answer/a563146) says that external application flows can report people who clicked to apply, rather than completed applications. Its [Recruiter reporting documentation](https://www.linkedin.com/help/linkedin/answer/a413441) separately defines views, apply clicks, and completed applications. We therefore use **recorded application count** as a cautious label for `applies`. Cross-flow differences may include measurement differences and must not be interpreted as causal effects on completed applications. These current product definitions identify a material risk; they do not establish the exact behavior of every historical API record.

## Missing values may include omitted zeros

At the inspected [collector commit](https://github.com/arshka/LinkedIn-Job-Scraper/commit/62c2e559234fe7b5cff3d902ccda9cea2d069c63), `clean_job_postings` retains mapped fields only when their values are truthy. Consequently, numeric zero and Boolean false are dropped by that implementation. See [the original cleaning code](https://github.com/arshka/LinkedIn-Job-Scraper/blob/62c2e559234fe7b5cff3d902ccda9cea2d069c63/scripts/helpers.py).

The recovered raw file contains:

- `23320` recorded `applies` values, all positive; `100529` missing values.
- `122160` recorded `views` values, all positive; `1689` missing values.
- `15246` recorded `remote_allowed` values, all `1`; the remainder missing.

There are `23319` records with both `applies` and `views` observed; one additional record has a known application count but missing views. Within the jointly observed records, the source audit found no `applies > views`. These are raw-data diagnostics, not final modeling sample sizes.

This pattern is consistent with omitted zeros and false values, but it does not prove that every missing entry means zero. The dataset includes later contributions whose precise collector revision was not established. We do not universally replace missing outcomes with zero or identify missing remote flags as verified onsite work.

Filtering to recorded outcomes creates a selected analysis population. Performance and associations estimated within it do not automatically generalize to postings without recorded outcomes. A positive-count Poisson model is a predictive working model here, not evidence that the unselected outcome follows a Poisson distribution.

## Exposure and time limitations

The original [database insertion code](https://github.com/arshka/LinkedIn-Job-Scraper/blob/62c2e559234fe7b5cff3d902ccda9cea2d069c63/scripts/database_scripts.py) stores a job capture timestamp in `scraped`. That field is absent from the recovered 31-column file. The separate company table's `time_recorded` does not establish each job's observation time.

There is no verified basis for treating `days_open * (views + 1)` as measured exposure. Views is already a recorded count, and an expiration interval can include time after those counts were captured. Replacing the missing observation endpoint with today's date also makes the feature change when the analysis is rerun.

The new primary specification estimates the coefficient of `log1p(views)` freely and uses zero offset. Fixed `log1p(views)` exposure and an expiration-interval specification are sensitivity analyses with explicit assumptions. They do not restore an observed person-time denominator. The resulting coefficients, feature importance, and model comparisons describe associations and predictions, not interventions.

## Citation and data rights

Suggested citation:

> Koneru, Arsh. (2024). *LinkedIn Job Postings (2023 - 2024)*, version 13 [Data set]. Kaggle. https://www.kaggle.com/datasets/arshkon/linkedin-job-postings. CC BY-SA 4.0. The data card additionally credits `@zoeyyuzou`.

CC BY-SA 4.0 permits redistribution and adaptation subject to its terms, including attribution, a license link, identification of modifications, and ShareAlike for distributed adaptations. It does not automatically clear unrelated third-party rights. The repository keeps the large raw data outside Git and provides [acquisition and verification instructions](../data/README.md). A license for newly written analysis code does not replace the source data license.

The collector code is cited as provenance evidence, not copied into the analysis pipeline. No license file was found in the inspected collector root; a public repository alone is not a license grant.

中文摘要：已找回并固定 Kaggle 第 13 版原始输入。空值可能混有采集器遗漏的零值，外站申请可能实际是点击数，且缺少逐岗位抓取时间。因此本项目属于对原始数据的重新分析，不能把旧结果当作已复现，也不能把模型关联解释为因果效果。
