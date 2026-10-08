# Corrections to the earlier project

The defensible outcome of this reconstruction is a new analysis of the recovered original data. The earlier presentation, report, working CSV, and scripts provide historical context; their numbers are not acceptance targets for the new pipeline.

## Input integrity

Two different files had the same base name. Inspection identified the following differences:

| Check | Recovered archive member | Earlier working CSV |
| --- | ---: | ---: |
| Bytes | 516843769 | 492354805 |
| Parsed records | 123849 | 124802 |
| Columns | 31 | 51 |
| Named columns | 31 | 18 |
| Missing `description` | 7 | 857 |
| Nonempty `applies` | 23320 | 23369 |
| Numeric `applies` | 23320 | 23307 |
| Extra exact duplicate records | 0 | 1271 |
| Implausible application-type entries under the audit's allowed-value check | 1 | 152 |

The working file also contains nonempty unnamed columns, timestamp-like strings in categorical fields, and implausible timestamp values. Its nonempty `applies` entries include 62 nonnumeric strings. This is structural damage, not a normal feature-selection step. The earlier missingness table matches the working file's counts. The precise editing or export operation that produced the damage is not established, so it is not attributed to a particular application or person.

A matched job record illustrates the displacement: the original has location `Joliet, IL`, `views = 27`, and work type `Full-time`. In the damaged file, the location field becomes `27`, the work-type field contains a timestamp-like number, and an application URL appears under the remote flag. This demonstrates a field-boundary failure rather than a meaningful new category.

The new input is the checksum-verified archive member matching the Kaggle v13 file size. See [Data source](DATA_SOURCE.md) for its hash, provenance, and remaining upstream limitations.

## Corrections applied to the analysis contract

| Earlier issue | Reconstruction rule |
| --- | --- |
| Damaged working CSV treated as raw data | Start from the recovered, verified 31-column source; validate schema and values before modeling. |
| IDs removed before validation | Retain the job identifier for duplicate and split-integrity checks; do not use it as a substantive predictor. |
| `days_open * (views + 1)` described as corrected exposure | Primary model uses a freely estimated `log1p(views)` effect and zero offset. Fixed views exposure and the expiration interval are labeled sensitivity specifications. |
| Expiry or the current execution date used as the observation endpoint | Do not claim these are capture times. Any expiration interval is a deterministic proxy with limited interpretation. |
| Missing `applies` excluded without discussing its mechanism | State that inference and evaluation concern the selected recorded-outcome subset. Upstream code may have omitted zeros; missing outcomes cannot all be proven to equal zero. |
| Missing `remote_allowed` read as reliable onsite work | Preserve uncertainty: the inspected collector omits Boolean false as well as absent values. |
| Application types treated as interchangeable completed-application outcomes | Use the cautious recorded-count label; acknowledge that offsite behavior can measure apply clicks. |
| Text features or preprocessing reused across held-out observations | Fit learned preprocessing on training data inside each evaluation split. |
| Models compared using inconsistent prediction sets | Compare all models on the same held-out observations and metrics. Label any training-set diagnostics separately. |
| Text effects, feature gain, or counterfactual ratios described as causal | Report model associations and predictive diagnostics. Do not infer that changing a posting feature causes the estimated change. |

These rules define the corrected analysis. Actual fitted results and sample counts must come from generated outputs, not manually entered values in this document.

## Why old reported values are not reused

The presentation and report contain different metric tables. In the presentation, a slide headed as Poisson also lists five fold-level metrics whose means instead match the later Lasso summary. Its XGBoost figure does not explicitly identify out-of-fold predictions, whereas the Lasso figure does. These artifacts alone cannot establish a fair comparison.

The report claims that XGBoost has the lowest RMSE, although its own table lists a lower RMSE for Lasso. Some reported effect directions conflict with the presentation's charts. The old filtering table also reports a retained population exactly twice the earlier nonmissing-outcome count; the table alone does not prove where this doubling arose.

Reusing old numbers would preserve these contradictions. This repository should be evaluated on traceable raw data, consistent preprocessing and splits, generated metrics, and honest limitations. It does not claim exact reproduction of an unknown historical execution state.

## Limits that code corrections cannot remove

- **Selection:** Missing outcomes dominate the raw file, and the mechanism is not fully identified.
- **Measurement:** Apply clicks and completed applications may differ by application flow.
- **Time:** Job-level capture times are absent from the recovered file; expiry is not a substitute.
- **Coverage:** This is an observational collection rather than a documented probability sample.
- **Interpretation:** A predictive count model does not identify a causal hiring or application-process effect.

The [source audit](DATA_SOURCE.md) links the original data card, collector code, and official platform definitions supporting these limitations.

中文摘要：本次修正以找回的原始 CSV 为起点，统一预处理和样本外评估，纠正旧文档的指标及解释口径。由于旧工作文件损坏、历史版本混用及采集测量限制，成果应称为“原始数据重新分析”，不应称为完全复刻旧结果。
