# Obtaining the data

Raw data is not committed to this repository. Download [LinkedIn Job Postings (2023 - 2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings), **version 13**, from its publisher. The analysis expects `data/raw/postings.csv`.

## Download and verify

From the repository root:

```sh
python -m pip install kagglehub
python scripts/download_data.py
```

The script requests only `postings.csv` from `arshkon/linkedin-job-postings/versions/13`. It verifies the file before placing it in `data/raw/`. An existing valid file is reused. A mismatched existing file causes an error and is not overwritten. Any account requirements are handled by Kaggle's official client; do not commit credentials.

The [official KaggleHub documentation](https://github.com/Kaggle/kagglehub#download-dataset) describes version-specific and single-file downloads.

If you already have the recovered source file, place it at the expected location and run:

```sh
python scripts/download_data.py --verify-only
```

To check a file elsewhere without copying or downloading it:

```sh
python scripts/download_data.py --verify-only --output /path/to/postings.csv
```

Expected identity:

```text
File:     postings.csv
Bytes:    516843769
SHA-256:  de33f27c8d1b226014e0cede003404e9b0acb66db718671fda9b3e3538eca6ea
Records:  123849
Columns:  31
```

The hash identifies the recovered raw input used for this reconstruction. The [Kaggle v13 manifest](https://www.kaggle.com/api/v1/datasets/list/arshkon/linkedin-job-postings?datasetVersionNumber=13) independently confirms the file size, but does not expose a hash. A future mismatch must be reviewed rather than silently accepted as the same input. Do not use an edited spreadsheet export solely because it is also named `postings.csv`.

The downloader verifies byte size and SHA-256. Schema, logical record count, missingness, and other analytical checks belong to the data-audit pipeline.

## Source license and interpretation

Publisher: Arsh Koneru; the data card additionally credits `@zoeyyuzou`. Source data is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), separately from newly written code. Retain attribution and license information when redistributing data, and document transformations.

See [Data source and measurement](../docs/DATA_SOURCE.md) before interpreting `applies`, `views`, remote-work flags, or time intervals. Missing values may include omitted zeros, and the observed-outcome subset is selected. This repository keeps raw data and derived record-level extracts outside version control and shares aggregate analysis outputs.
