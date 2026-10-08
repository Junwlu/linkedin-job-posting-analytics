"""Regression checks for the concrete defects found in the historical files."""
from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from linkedin_analytics.data import load_and_clean


def row(job_id, applies="2", views="10", **overrides):
    record = dict(job_id=str(job_id), title="Senior Data Analyst", description="SQL\nPython, reporting",
                  applies=applies, views=views, company_id="123.0", company_name="Example",
                  location="Chicago, IL", application_type="SimpleOnsiteApply",
                  formatted_work_type="Full-time", formatted_experience_level="Mid-Senior level",
                  remote_allowed="1.0", sponsored="0", normalized_salary="80000", currency="USD",
                  listed_time="1710000000000", original_listed_time="1710000000000",
                  expiry="1712592000000", closed_time="")
    record.update(overrides)
    return record


class DataCleaningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "postings.csv"

    def write(self, records):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        return self.path

    def test_outcome_missing_is_not_zero_and_views_stay_missing(self):
        df, audit = load_and_clean(self.write([
            row(1, applies=""), row(2, applies="0"), row(3, applies="-1"),
            row(4, applies="2.5"), row(5, applies="Full-time"), row(6, views=""),
            row(7, views="-1"), row(8, views="0"), row(9, views="1.5"),
        ]))
        self.assertEqual(df.row_id.tolist(), ["2", "8"])
        self.assertEqual(audit["raw_applies_missing"], 1)
        self.assertEqual(audit["raw_applies_invalid_nonmissing"], 3)
        self.assertEqual(audit["main_model_rows"], 2)
        self.assertEqual(audit["observed_applies_zero"], 1)
        self.assertEqual(audit["views_zero_applies_positive"], 1)
        self.assertEqual(audit["observed_applies_missing_or_invalid_views"], 3)
        self.assertEqual(audit["clean_observed_count_rows"], 5)
        self.assertTrue(df.valid_views.all())
        self.assertEqual(float(df.loc[df.row_id.eq("8"), "offset"].iloc[0]), 0.0)

    def test_boolean_mapping_is_single_pass_and_unknown_categories_do_not_drop_rows(self):
        df, _ = load_and_clean(self.write([
            row(1, remote_allowed="1.0", sponsored="0"),
            row(2, remote_allowed="0", sponsored="1"),
            row(3, remote_allowed="", sponsored="", application_type="1.71286E+12"),
        ]))
        self.assertEqual(df.remote_allowed.tolist(), ["Remote", "Onsite", "Unknown"])
        self.assertEqual(df.sponsored.tolist(), ["NotSponsored", "Sponsored", "Unknown"])
        self.assertEqual(df.application_type.iloc[2], "Unknown")
        self.assertEqual(len(df), 3)

    def test_expiry_does_not_enter_primary_offset_or_use_today(self):
        df, _ = load_and_clean(self.write([
            row(1), row(2, expiry="1725552000000"),
            row(3, expiry="1700000000000"), row(4, expiry="", listed_time=""),
            row(5, closed_time="1710043200000"),
        ]))
        np.testing.assert_allclose(df.offset, np.log1p(10))
        self.assertAlmostEqual(df.legacy_days_open.iloc[0], 30)
        self.assertAlmostEqual(df.legacy_days_open.iloc[1], 180)
        self.assertFalse(df.legacy_days_valid.iloc[2])
        self.assertFalse(df.legacy_days_valid.iloc[3])
        self.assertEqual(df.legacy_days_open.iloc[4], 1)
        self.assertEqual(str(df.listed_time_utc.dt.tz), "UTC")

    def test_zip_and_plain_csv_are_identical_and_payload_fingerprinted(self):
        self.write([row(1)])
        zipped = self.path.with_suffix(".csv.zip")
        with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.path, "postings.csv")
        plain_df, plain_audit = load_and_clean(self.path)
        zip_df, zip_audit = load_and_clean(zipped)
        pd.testing.assert_frame_equal(plain_df, zip_df)
        self.assertEqual(zip_audit["raw_dataset_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(zip_audit["raw_dataset_sha256"], plain_audit["raw_dataset_sha256"])
        self.assertNotEqual(zip_audit["source_sha256"], plain_audit["source_sha256"])

    def test_duplicate_grain_and_damaged_header_are_rejected(self):
        self.write([row(1), row(1, applies="3")])
        with self.assertRaisesRegex(ValueError, "Conflicting repeated job_id"):
            load_and_clean(self.path)
        self.path.write_text("company_name,title,description,applies,views,,\nExample,T,D,1,2,,\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Blank/duplicate"):
            load_and_clean(self.path)

    def test_salary_currency_is_explicit_without_discarding_posting(self):
        df, audit = load_and_clean(self.write([row(1), row(2, currency="CAD"), row(3, normalized_salary="-1")]))
        self.assertEqual(len(df), 3)
        self.assertEqual(df.normalized_salary.iloc[0], 80000)
        self.assertTrue(df.normalized_salary.iloc[1:].isna().all())
        self.assertEqual(df.normalized_salary_raw.iloc[1], 80000)
        self.assertEqual(audit["observed_salary_values_excluded_for_currency_or_validity"], 2)


if __name__ == "__main__":
    unittest.main()
