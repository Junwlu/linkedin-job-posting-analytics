"""Regression tests for train/test leakage and silent row-loss defects."""
import unittest

import numpy as np
import pandas as pd

from linkedin_analytics.features import FeatureBuilder


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "title": ["Data analyst", "Software engineer", "Nurse", "Sales manager"] * 3,
            "description": ["python sql analysis", "cloud software engineering", "patient nursing care", "sales retail marketing"] * 3,
            "views": np.arange(1, 13), "normalized_salary": np.arange(40_000, 52_000, 1000),
            "application_type": ["OffsiteApply", "SimpleOnsiteApply"] * 6,
            "formatted_work_type": ["Full-time"] * 12,
            "remote_allowed": ["Remote", "Unknown"] * 6,
            "sponsored": ["NotSponsored"] * 12,
        })

    def test_holdout_never_changes_vocabulary_or_salary_cutoffs(self):
        builder = FeatureBuilder(n_svd=2, min_category_count=1, min_text_df=1)
        builder.fit(self.train)
        before = builder.diagnostics()
        test = self.train.iloc[:2].copy()
        test["description"] = ["heldoutuniquetoken", "anotherunseenword"]
        test["normalized_salary"] = 1e12
        matrix = builder.transform(test)
        self.assertEqual(matrix.shape[0], 2)
        self.assertNotIn("heldoutuniquetoken", builder.vectorizer_.vocabulary_)
        self.assertEqual(builder.salary_cutoffs_, np.quantile(self.train["normalized_salary"], [.33, .67]).tolist())
        self.assertEqual(before, builder.diagnostics())

    def test_reference_category_and_unseen_levels_preserve_rows(self):
        builder = FeatureBuilder(include_svd=False, min_category_count=1).fit(self.train)
        self.assertEqual(builder.category_maps_["application_type"]["reference"], "OffsiteApply")
        self.assertFalse(builder.category_maps_["remote_allowed"]["reference_fallback"])
        self.assertEqual(builder.category_maps_["remote_allowed"]["reference"], "Unknown")
        self.assertNotIn("Onsite", builder.category_maps_["remote_allowed"]["levels"])
        test = self.train.iloc[:2].copy()
        test["application_type"] = ["never-seen", None]
        matrix = builder.transform(test)
        self.assertEqual(matrix.shape, (2, len(builder.feature_names_)))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(builder.diagnostics(test)["heldout_category_fallback_counts"]["application_type"], 2)

    def test_constant_sponsored_and_redundant_constructs_are_absent(self):
        builder = FeatureBuilder(include_svd=False, min_category_count=1).fit(self.train)
        self.assertFalse(any(name.startswith("sponsored__") for name in builder.feature_names_))
        self.assertNotIn("skill_hits", builder.feature_names_)
        self.assertNotIn("is_intern", builder.feature_names_)
        self.assertIn("log1p_views", builder.feature_names_)
        offset_builder = FeatureBuilder(include_svd=False, include_log_views=False, min_category_count=1).fit(self.train)
        self.assertNotIn("log1p_views", offset_builder.feature_names_)

    def test_rare_mapping_is_learned_from_train_only(self):
        train = self.train.copy()
        train.loc[train.index[0], "application_type"] = "RareApply"
        builder = FeatureBuilder(include_svd=False, min_category_count=2).fit(train)
        self.assertNotIn("RareApply", builder.category_maps_["application_type"]["kept"])
        self.assertIn("__Other__", builder.category_maps_["application_type"]["levels"])
        test = self.train.iloc[:3].copy()
        test["application_type"] = "RareApply"
        matrix = builder.transform(test)
        index = builder.feature_names_.index("application_type____Other__")
        np.testing.assert_array_equal(matrix[:, index], np.ones(3))

    def test_missing_views_raise_instead_of_silent_zero(self):
        test = self.train.copy()
        test.loc[0, "views"] = np.nan
        with self.assertRaises(ValueError):
            FeatureBuilder(include_svd=False).fit(test)


if __name__ == "__main__":
    unittest.main()
