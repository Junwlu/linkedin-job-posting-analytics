"""Load source postings without repeating the damaged spreadsheet conversion.

The model population is conditional on an observed application count. Missing
counts are never zeros. The primary model has offset zero and estimates a
coefficient for log1p(views). Returned ``exposure``/``offset`` columns are reach
proxy sensitivity inputs, not person-time or binomial trials. Expiry is used
only in a named legacy sensitivity.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

ORIGINAL_CSV_SHA256 = "de33f27c8d1b226014e0cede003404e9b0acb66db718671fda9b3e3538eca6ea"
REQUIRED_COLUMNS = {"job_id", "title", "description", "applies", "views"}
DATE_COLUMNS = ("original_listed_time", "listed_time", "expiry", "closed_time")
CATEGORY_VALUES = {
    "application_type": {"ComplexOnsiteApply", "SimpleOnsiteApply", "OffsiteApply"},
    "formatted_work_type": {"Full-time", "Part-time", "Contract", "Temporary", "Internship", "Volunteer", "Other"},
    "formatted_experience_level": {"Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable"},
}
MISSING_TOKENS = {"", "nan", "na", "n/a", "null", "none"}
REGIONS = {
    "Northeast": set("CT ME MA NH RI VT NJ NY PA".split()),
    "Midwest": set("IL IN IA KS MI MN MO NE ND OH SD WI".split()),
    "South": set("AL AR DE DC FL GA KY LA MD MS NC OK SC TN TX VA WV".split()),
    "West": set("AK AZ CA CO HI ID MT NM NV OR UT WA WY".split()),
}
SENIORITY_PATTERNS = (
    ("cxo", r"\b(chief|c[eo]o|cto|cfo)\b"),
    ("vp", r"\b(vp|vice president)\b"),
    ("director", r"\bdirector\b"),
    ("lead", r"\b(lead|principal)\b"),
    ("senior", r"\b(senior|sr)\b"),
    ("associate", r"\bassociate\b"),
    ("junior", r"\b(junior|jr)\b"),
    ("intern", r"\b(intern|internship)\b"),
)
ROLE_PATTERNS = (
    ("Data", r"\b(data|analytics?|bi|business intelligence|scientist)\b"),
    ("Analytics", r"\b(analyst|analytics?)\b"),
    ("Engineering", r"\b(engineer|developer|software|full[- ]?stack)\b"),
    ("Cloud", r"\b(cloud|aws|azure|gcp|devops?)\b"),
    ("Sales", r"\b(sales?|business development|bd)\b"),
    ("Marketing", r"\b(marketing|growth|seo)\b"),
    ("Nursing", r"\b(nurse|rn|registered nurse)\b"),
)


def _sha256(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


@contextmanager
def _csv_stream(path: Path, member: str | None) -> Iterator:
    if member is None:
        with path.open("rb") as stream:
            yield stream
    else:
        with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
            yield stream


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _missing(series: pd.Series) -> pd.Series:
    return _text(series).str.lower().isin(MISSING_TOKENS)


def _numbers(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(np.isfinite(values))


def _valid_counts(values: pd.Series) -> pd.Series:
    return values.notna() & values.ge(0) & values.eq(np.floor(values))


def _tri_state(series: pd.Series, true_label: str, false_label: str) -> pd.Series:
    # Map exactly once from original values. Missing is never interpreted as 0.
    mapping = {"1": true_label, "1.0": true_label, "true": true_label,
               "0": false_label, "0.0": false_label, "false": false_label}
    return _text(series).str.lower().map(mapping).fillna("Unknown")


def _region(location: str) -> str:
    text = str(location).upper().strip()
    if not text or text.lower() in MISSING_TOKENS:
        return "Unknown"
    if "REMOTE" in text:
        return "Remote"
    if text in {"UNITED STATES", "USA", "US"} or text.endswith(", US"):
        return "US-Unknown"
    match = re.search(r",\s*([A-Z]{2})(?:\s|$)", text)
    if match:
        for region, states in REGIONS.items():
            if match.group(1) in states:
                return region
    # Unrecognized locations are not automatically assigned to the US.
    return "Other"


def _title_category(title: str, patterns: tuple, fallback: str) -> str:
    for label, pattern in patterns:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return label
    return fallback


def load_and_clean(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    """Return the common valid-count/valid-views cohort plus a data audit.

    Accept an original UTF-8 CSV or ZIP containing one ``postings.csv``. Invalid
    outcome/identifier rows are counted and excluded; invalid feature values
    become missing or Unknown. Missing/invalid views are counted before the
    final common-cohort filter. No input is modified or extracted to disk.
    """
    path = Path(raw_path)
    member = None
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if Path(n).name == "postings.csv"]
            if len(names) != 1:
                raise ValueError("ZIP must contain exactly one member named postings.csv")
            member = names[0]
    with path.open("rb") as stream:
        container_sha = _sha256(stream)
    with _csv_stream(path, member) as stream:
        csv_sha = _sha256(stream)
    with _csv_stream(path, member) as stream:
        wrapper = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
        header = next(csv.reader(wrapper))
    if len(header) != len(set(header)) or any(not c for c in header):
        raise ValueError("Blank/duplicate CSV columns indicate a damaged or non-original source")
    absent = REQUIRED_COLUMNS.difference(header)
    if absent:
        raise ValueError(f"Required raw columns missing: {sorted(absent)}; use original postings.csv")

    audit = {
        "source_name": path.name,
        "zip_member": member,
        "source_sha256": container_sha,
        "raw_dataset_sha256": csv_sha,
        "matches_original_v13_fingerprint": csv_sha == ORIGINAL_CSV_SHA256,
        "raw_columns": header,
        "raw_column_count": len(header),
        "has_scrape_timestamp": any("scrap" in c.lower() for c in header),
        "raw_rows": 0,
        "raw_applies_missing": 0,
        "raw_applies_invalid_nonmissing": 0,
        "raw_views_missing": 0,
        "raw_views_invalid_nonmissing": 0,
        "invalid_or_missing_job_id_rows": 0,
    }
    raw_ids = Counter()
    observed_by_application = Counter()
    total_by_application = Counter()
    missing_counts = Counter()
    collected = []
    with _csv_stream(path, member) as stream:
        for chunk in pd.read_csv(stream, encoding="utf-8-sig", dtype=str,
                                 keep_default_na=False, chunksize=20000,
                                 on_bad_lines="error"):
            audit["raw_rows"] += len(chunk)
            for column in chunk:
                missing_counts[column] += int(_missing(chunk[column]).sum())
            applies = _numbers(chunk["applies"])
            valid_applies = _valid_counts(applies)
            views = _numbers(chunk["views"])
            valid_views = _valid_counts(views)
            miss_applies = _missing(chunk["applies"])
            miss_views = _missing(chunk["views"])
            audit["raw_applies_missing"] += int(miss_applies.sum())
            audit["raw_applies_invalid_nonmissing"] += int((~valid_applies & ~miss_applies).sum())
            audit["raw_views_missing"] += int(miss_views.sum())
            audit["raw_views_invalid_nonmissing"] += int((~valid_views & ~miss_views).sum())
            ids = _text(chunk["job_id"]).str.replace(r"\.0$", "", regex=True)
            valid_id = ids.str.fullmatch(r"[0-9]+", na=False)
            audit["invalid_or_missing_job_id_rows"] += int((~valid_id).sum())
            raw_ids.update(ids[valid_id])
            application = _text(chunk.get("application_type", pd.Series("", index=chunk.index)))
            total_by_application.update(application)
            observed_by_application.update(application[valid_applies])
            keep = valid_applies & valid_id
            selected = chunk.loc[keep].copy()
            selected["job_id"] = ids[keep]
            selected["applies"] = applies[keep].astype(np.int64)
            selected["views"] = views[keep].where(valid_views[keep])
            collected.append(selected)
    audit["raw_missing_by_column"] = dict(missing_counts)
    audit["raw_unique_valid_job_ids"] = len(raw_ids)
    audit["raw_duplicate_job_id_extra_rows"] = sum(n - 1 for n in raw_ids.values())
    audit["applies_observability_by_application_type"] = {
        label: {"raw_rows": count, "observed_applies": observed_by_application[label],
                "observed_fraction": observed_by_application[label] / count}
        for label, count in sorted(total_by_application.items())
    }
    if not collected:
        raise ValueError("No readable CSV records")
    df = pd.concat(collected, ignore_index=True)
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    audit["observed_rows_exact_duplicates_removed"] = before_dedup - len(df)
    if df["job_id"].duplicated().any():
        raise ValueError("Conflicting repeated job_id records: resolve source grain before modeling")
    if df.empty:
        raise ValueError("No valid observed application counts with valid identifiers")
    df["row_id"] = df["job_id"].astype(str)
    for column in ["title", "description", "company_name", "location", "company_id", "currency", "normalized_salary",
                   "remote_allowed", "sponsored", *CATEGORY_VALUES, *DATE_COLUMNS]:
        if column not in df:
            df[column] = ""
    for column in ["title", "description", "company_name", "location"]:
        df[column] = df[column].fillna("").astype(str)
    df["company_id"] = _text(df["company_id"]).str.replace(r"\.0$", "", regex=True)
    df["company_id"] = df["company_id"].mask(_missing(df["company_id"]), pd.NA)

    audit["category_unknown_counts"] = {}
    for column, allowed in CATEGORY_VALUES.items():
        df[column + "_raw"] = df[column]
        labels = _text(df[column])
        df[column] = labels.where(labels.isin(allowed), "Unknown")
        audit["category_unknown_counts"][column] = int(df[column].eq("Unknown").sum())
    for column, positive, negative in [("remote_allowed", "Remote", "Onsite"),
                                       ("sponsored", "Sponsored", "NotSponsored")]:
        df[column + "_raw"] = df[column]
        df[column] = _tri_state(df[column], positive, negative)
        audit["category_unknown_counts"][column] = int(df[column].eq("Unknown").sum())

    salary = _numbers(df["normalized_salary"])
    df["normalized_salary_raw"] = salary
    salary_usable = salary.ge(0) & _text(df["currency"]).eq("USD")
    df["normalized_salary"] = salary.where(salary_usable)
    audit["observed_salary_values_excluded_for_currency_or_validity"] = int((salary.notna() & ~salary_usable).sum())
    df["region"] = df["location"].map(_region)
    df["seniority"] = df["title"].map(lambda title: _title_category(title, SENIORITY_PATTERNS, "Unknown"))
    df["role_family"] = df["title"].map(lambda title: _title_category(title, ROLE_PATTERNS, "Other"))
    df["valid_views"] = _valid_counts(df["views"])
    df["exposure"] = (1.0 + df["views"]).where(df["valid_views"])
    df["offset"] = np.log(df["exposure"])
    for column in DATE_COLUMNS:
        df[column + "_utc"] = pd.to_datetime(_numbers(df[column]), unit="ms", utc=True, errors="coerce")

    # This planned/end-window proxy exists only to compare with the old report.
    closed, expiry = df["closed_time_utc"], df["expiry_utc"]
    end = closed.where(expiry.isna() | (closed.notna() & closed.le(expiry)), expiry)
    gap = (end - df["listed_time_utc"]).dt.total_seconds() / 86400.0
    df["legacy_days_valid"] = gap.notna() & gap.gt(0) & np.isfinite(gap)
    df["legacy_days_unclipped"] = gap.where(df["legacy_days_valid"])
    df["legacy_days_open"] = gap.where(df["legacy_days_valid"]).clip(lower=1.0)
    audit.update({
        "clean_observed_count_rows": len(df),
        "main_model_rows": int(df["valid_views"].sum()),
        "observed_applies_missing_or_invalid_views": int((~df["valid_views"]).sum()),
        "observed_applies_zero": int(df["applies"].eq(0).sum()),
        "views_zero_applies_positive": int((df["views"].eq(0) & df["applies"].gt(0)).sum()),
        "applies_greater_than_views": int((df["applies"] > df["views"]).sum()),
        "legacy_days_valid_rows": int(df["legacy_days_valid"].sum()),
        "legacy_days_clipped_to_one": int((df["legacy_days_valid"] & gap.lt(1)).sum()),
        "main_offset_definition": "0; log1p(views) enters the primary model as a covariate with an estimated coefficient",
        "reach_proxy_offset_definition": "log(1 + views); sensitivity-only reach proxy, not person-time or binomial trials",
        "legacy_window_definition": "max((min(valid closed_time, valid expiry) - listed_time) / 86400000, 1), only positive valid intervals; planned/end-window sensitivity proxy, not observed time at risk",
        "population_limitation": "Conditioned on observed application counts. Missing is not zero; all observed outcomes in the pinned source are positive. No scrape timestamp is available to establish observation duration.",
        "row_id_unique": bool(df["row_id"].is_unique),
        "retained_category_counts": {c: {str(k): int(v) for k, v in df[c].value_counts().items()}
                                     for c in [*CATEGORY_VALUES, "remote_allowed", "sponsored"]},
    })
    main = df.loc[df["valid_views"]].reset_index(drop=True)
    audit["exclusion_flow"] = [
        {"stage": "raw_source", "rows": audit["raw_rows"]},
        {"stage": "valid_observed_application_count", "rows": audit["raw_rows"] - audit["raw_applies_missing"] - audit["raw_applies_invalid_nonmissing"]},
        {"stage": "valid_identifier_before_deduplication", "rows": before_dedup},
        {"stage": "unique_observed_count_records", "rows": len(df)},
        {"stage": "common_valid_views_cohort", "rows": len(main)},
    ]
    audit["returned_rows"] = len(main)
    audit["main_category_counts"] = {c: {str(k): int(v) for k, v in main[c].value_counts().items()}
                                      for c in [*CATEGORY_VALUES, "remote_allowed", "sponsored"]}
    return main, audit
