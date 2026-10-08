# Resume-ready project description

These bullets describe the reconstructed analysis. The course project ran in October–December 2025; the source audit, corrected pipeline and new performance figures were completed in October 2026. Use dates that accurately reflect which version you are presenting.

## 中文

**LinkedIn 岗位申请数据分析｜Python、统计建模与文本分析**

- 使用 Python 重建 Kaggle LinkedIn 岗位数据分析流程，审计 123,849 条原始记录，识别文件损坏与申请数缺失问题，形成 23,319 条有记录申请数和浏览量的可追溯分析样本。
- 整合岗位属性、标题关键词与 TF-IDF/SVD 文本特征，在统一五折交叉验证下比较 Poisson、L1 Poisson 和 XGBoost；XGBoost 的 MAE 达到 4.11，较 Poisson 模型降低 17.7%。
- 通过曝光假设敏感性检验、公司聚类稳健区间及独立指标复算校正结论，交付包含数据校验、自动化分析、已执行 Notebook 和报告的可复现 GitHub 项目。

## English

**LinkedIn Job Posting Analytics | Python, Statistical Modeling & NLP**

- Rebuilt a Python analysis pipeline for 123,849 Kaggle LinkedIn postings, audited source-file corruption and outcome missingness, and established a traceable cohort of 23,319 postings with recorded application and view counts.
- Engineered structured, title-keyword and TF-IDF/SVD features and compared Poisson, L1-Poisson and XGBoost models using shared five-fold validation; achieved XGBoost MAE of 4.11, 17.7% below the Poisson model.
- Audited exposure assumptions and conditional associations using sensitivity analyses, company-cluster robust intervals and independent metric checks; published a reproducible GitHub pipeline with data validation, an executed notebook and a corrected report.

## Evidence and wording boundaries

Performance figures come from `results/cv_summary.csv`; sample counts come from `results/data_audit.json`. The percentage reduction compares models in the new run, not the old report. The workflow did not measure a business increase in applications.

Use Python for the maintained reconstruction. R can be discussed as part of the original coursework, but the new pipeline is not written in R. No recovered SQL implementation supports listing SQL as an implemented project tool. The SQL title indicator is a data feature, not evidence of SQL programming.

Avoid the earlier general claims that full-time positions consistently outperform other work types or that all explicit technical signals imply stronger response. Specific title-keyword and application-flow associations are observational, conditional on views and selected recorded outcomes, and subject to measurement differences.
