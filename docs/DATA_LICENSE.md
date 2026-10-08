# Code, data, and report licenses

The repository contains materials with different licenses. The root [MIT license](../LICENSE) applies to the newly written analysis software; it does not relicense the Kaggle dataset or the analytical outputs described below.

| Material | License and scope |
| --- | --- |
| New source code, scripts, tests, workflow configuration, and notebook code cells | MIT, copyright 2026 Junwei Lu. General software usage documentation is also MIT unless identified below as analytical content. |
| Original `postings.csv` and source-data excerpts | The publisher declares CC BY-SA 4.0. Attribution remains with the source creators, not the analysis author. The large input is acquired separately and is not distributed in Git. |
| Cleaned or transformed data, prediction tables, fitted model artifacts, result tables, figures, and new analytical report text released by this project | CC BY-SA 4.0 for the project's copyrightable contributions, copyright 2026 Junwei Lu, with the upstream data attribution retained. This includes analytical notebook narrative and outputs; notebook code remains MIT. |
| Earlier coursework reports, slides, working files, and other recovered historical materials | Not part of this repository's license grant. They are provenance evidence, and this project does not assert permission to redistribute third-party or jointly authored content within them. |
| Third-party software and quoted material | Their respective notices and licenses continue to apply. |

This scope identifies the project's license choices; it does not assert that every numerical fact, fitted parameter, or individual record is independently protected by copyright. Existing public-domain rights and applicable exceptions remain unaffected.

## Dataset attribution

> Koneru, Arsh. (2024). *LinkedIn Job Postings (2023 - 2024)*, version 13 [Data set]. Kaggle. https://www.kaggle.com/datasets/arshkon/linkedin-job-postings. CC BY-SA 4.0. The data card additionally credits `@zoeyyuzou` for data collection.

Source: [Kaggle dataset and data card](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings). License: [CC BY-SA 4.0 summary](https://creativecommons.org/licenses/by-sa/4.0/) and [legal terms](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en). Version, file fingerprint, collector evidence, and measurement limits are documented in [DATA_SOURCE.md](DATA_SOURCE.md).

The project changes the source by selecting records with observed valid counts and views, validating categories and identifiers, creating features, fitting models, and producing analytical summaries. These transformations do not alter the original input file. [CORRECTIONS.md](CORRECTIONS.md) describes the departure from the earlier working file and analysis.

## Reuse and attribution

When distributing covered data or adaptations, retain appropriate source attribution, link the license, identify modifications, and apply the required ShareAlike terms to adaptations. The license permits commercial reuse as well as noncommercial reuse; this project adds no noncommercial restriction. Its grant does not establish ownership of unrelated third-party rights or imply endorsement by the data creators, Kaggle, or LinkedIn.

A suitable credit for a reused project figure or result table is:

> Analysis: Junwei Lu (2026), *LinkedIn Job Posting Analytics*, CC BY-SA 4.0. Source data: Arsh Koneru, *LinkedIn Job Postings (2023 - 2024)*, Kaggle version 13, CC BY-SA 4.0; additional collector credited by the data card: `@zoeyyuzou`. Changes: [describe any further changes].

Include the [project link](https://github.com/Junwlu/linkedin-job-posting-analytics), [dataset link](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings), and [license link](https://creativecommons.org/licenses/by-sa/4.0/) with that credit. The software citation and dataset citation are recorded separately in [CITATION.cff](../CITATION.cff).

The upstream scraper is linked for provenance, not incorporated as software in this repository. No new license is assigned to its code by this project.

中文说明：新写的程序代码使用 MIT；原始数据和本项目发布的数据衍生结果、图表及分析报告使用 CC BY-SA 4.0，并保留原数据署名。找回的旧课业文件不因本项目发布而自动获得新的授权。
