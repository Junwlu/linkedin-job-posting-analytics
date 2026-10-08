"""Training-only feature construction for the LinkedIn posting analysis.

Pure cleaning lives in data.py. This module learns vocabulary, salary cutoffs,
rare levels and feature availability from its fit rows only. It never drops rows.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


KEYWORDS = (
    "data", "analyst", "engineer", "developer", "manager", "nurse", "sales",
    "marketing", "python", "sql", "java", "cloud", "aws", "azure", "gcp",
)
CATEGORY_REFERENCES = {
    "application_type": "OffsiteApply",
    "formatted_work_type": "Full-time",
    "formatted_experience_level": "Mid-Senior level",
    "remote_allowed": "Unknown",
    "sponsored": "NotSponsored",
    "region": "South",
    "seniority": "Unknown",
    "role_family": "Other",
    "salary_band": "Med",
}
RARE_LEVEL = "__Other__"


def _strings(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="string")
    return df[column].astype("string").fillna(default)


def _number(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _feature_name(column: str, level: str) -> str:
    # XGBoost feature names cannot contain [, ] or <.
    return column + "__" + re.sub(r"[^a-zA-Z0-9_+.-]", "_", str(level))


@dataclass
class FeatureConfig:
    n_svd: int = 15
    max_text_features: int = 1500
    min_category_count: int = 20
    min_text_df: int = 2
    random_state: int = 20251201
    include_svd: bool = True
    include_log_views: bool = True


class FeatureBuilder:
    """A serializable fit/transform object with explicit category references.

    Numeric lengths have interpretable units. GLM scaling, when requested, is
    performed inside the estimator using training data, and reported coefficients
    remain on these original feature scales.
    """

    def __init__(self, config: FeatureConfig | None = None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass a FeatureConfig or keyword options, not both.")
        self.config = config or FeatureConfig(**kwargs)

    def _categories(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {
            key: _strings(df, key, "Unknown").to_numpy(dtype=str)
            for key in CATEGORY_REFERENCES if key != "salary_band"
        }
        salary = _number(df, "normalized_salary")
        valid = np.isfinite(salary) & (salary >= 0)
        band = np.full(len(df), "Unknown", dtype=object)
        if self.salary_cutoffs_ is not None:
            lo, hi = self.salary_cutoffs_
            band[valid] = np.where(salary[valid] <= lo, "Low",
                                   np.where(salary[valid] <= hi, "Med", "High"))
        out["salary_band"] = band
        return out

    def _numeric(self, df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
        title = _strings(df, "title")
        description = _strings(df, "description")
        values = [title.str.len().to_numpy(float) / 10,
                  description.str.len().to_numpy(float) / 1000]
        metadata = [
            {"name": "title_len_10", "kind": "numeric", "unit": "10 title characters"},
            {"name": "desc_len_1000", "kind": "numeric", "unit": "1000 description characters"},
        ]
        lowered = title.str.lower()
        for keyword in KEYWORDS:
            values.append(lowered.str.contains(
                rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", regex=True
            ).to_numpy(float))
            metadata.append({"name": f"kw_{keyword}", "kind": "keyword", "unit": "0 to 1 in title"})
        if self.config.include_log_views:
            views = _number(df, "views")
            if not np.all(np.isfinite(views) & (views >= 0)):
                raise ValueError("FeatureBuilder requires observed nonnegative views.")
            values.append(np.log1p(views))
            metadata.append({"name": "log1p_views", "kind": "numeric", "unit": "log(1 + views)"})
        return np.column_stack(values), metadata

    def _matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
        blocks, metadata = [], []
        for column, values in self._categories(df).items():
            info = self.category_maps_[column]
            fallback = RARE_LEVEL if RARE_LEVEL in info["levels"] else info["reference"]
            mapped = np.array([v if v in info["kept"] else fallback for v in values], dtype=object)
            for level in info["levels"]:
                if level == info["reference"]:
                    continue
                blocks.append((mapped == level).astype(float)[:, None])
                metadata.append({
                    "name": _feature_name(column, level), "kind": "category",
                    "column": column, "level": str(level), "reference": info["reference"],
                    "unit": f"{level} vs {info['reference']}",
                })
        numerical, numerical_meta = self._numeric(df)
        blocks.append(numerical)
        metadata.extend(numerical_meta)
        if self.vectorizer_ is not None and self.svd_ is not None:
            text = self.vectorizer_.transform(_strings(df, "description").tolist())
            dense_text = self.svd_.transform(text)
            blocks.append(dense_text)
            metadata.extend({"name": f"desc_svd_{i + 1}", "kind": "svd", "unit": "one SVD score unit"}
                            for i in range(dense_text.shape[1]))
        matrix = np.column_stack(blocks)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite feature matrix; no rows were silently removed.")
        return matrix, metadata

    def fit(self, df: pd.DataFrame, y=None):
        if len(df) < 2:
            raise ValueError("At least two training rows are required.")
        salary = _number(df, "normalized_salary")
        salary = salary[np.isfinite(salary) & (salary >= 0)]
        self.salary_cutoffs_ = np.quantile(salary, [.33, .67]).tolist() if len(salary) else None
        self.category_maps_ = {}
        for column, values in self._categories(df).items():
            counts = pd.Series(values).value_counts()
            preferred = CATEGORY_REFERENCES[column]
            kept = set(counts[counts >= self.config.min_category_count].index.astype(str))
            if preferred in counts:
                kept.add(preferred)
            mapped = pd.Series([v if v in kept else RARE_LEVEL for v in values])
            observed_counts = mapped.value_counts()
            levels = sorted(observed_counts.index.astype(str))
            reference = preferred if preferred in levels else str(observed_counts.index[0])
            self.category_maps_[column] = {
                "kept": sorted(kept), "levels": levels, "reference": reference,
                "preferred_reference": preferred,
                "reference_fallback": reference != preferred,
                "training_counts": {str(k): int(v) for k, v in counts.items()},
            }
        self.vectorizer_, self.svd_ = None, None
        self.text_note_ = "SVD disabled by configuration."
        if self.config.include_svd:
            vectorizer = TfidfVectorizer(
                max_features=self.config.max_text_features, ngram_range=(1, 2),
                stop_words="english", min_df=self.config.min_text_df,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                dtype=np.float64,
            )
            try:
                text = vectorizer.fit_transform(_strings(df, "description").tolist())
                n_components = min(self.config.n_svd, text.shape[1] - 1, len(df) - 1)
                if n_components > 0:
                    svd = TruncatedSVD(n_components=n_components, random_state=self.config.random_state)
                    svd.fit(text)
                    self.vectorizer_, self.svd_ = vectorizer, svd
                    self.text_note_ = f"{n_components} SVD components learned on training descriptions."
                else:
                    self.text_note_ = "Insufficient training vocabulary for SVD; no SVD columns."
            except ValueError as exc:
                if "empty vocabulary" not in str(exc) and "After pruning" not in str(exc):
                    raise
                self.text_note_ = "Empty training vocabulary after pruning; no SVD columns."
        matrix, metadata = self._matrix(df)
        self.variable_indices_ = np.flatnonzero(np.std(matrix, axis=0) > 1e-12)
        self.feature_metadata_ = [metadata[i] for i in self.variable_indices_]
        self.feature_names_ = [item["name"] for item in self.feature_metadata_]
        if len(set(self.feature_names_)) != len(self.feature_names_):
            raise ValueError("Feature-name collision after category name sanitization.")
        self.dropped_constant_features_ = [
            metadata[i]["name"] for i in range(matrix.shape[1]) if i not in set(self.variable_indices_)
        ]
        self.n_training_rows_ = len(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "variable_indices_"):
            raise RuntimeError("Call fit before transform.")
        matrix, _ = self._matrix(df)
        return np.asfortranarray(matrix[:, self.variable_indices_], dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(df, y).transform(df)

    def diagnostics(self, df: pd.DataFrame | None = None) -> dict:
        vocabulary = sorted(self.vectorizer_.vocabulary_) if self.vectorizer_ is not None else []
        result = {
            "n_training_rows": self.n_training_rows_, "n_features": len(self.feature_names_),
            "salary_cutoffs": self.salary_cutoffs_, "categories": self.category_maps_,
            "feature_metadata": self.feature_metadata_, "dropped_constant_features": self.dropped_constant_features_,
            "text_note": self.text_note_, "vocabulary_size": len(vocabulary),
            "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
        }
        if df is not None:
            result["heldout_category_fallback_counts"] = {
                column: int(sum(v not in self.category_maps_[column]["kept"] for v in values))
                for column, values in self._categories(df).items()
            }
        return result

    def topic_terms(self, n_terms: int = 10) -> list[dict]:
        if self.svd_ is None:
            return []
        words = self.vectorizer_.get_feature_names_out()
        output = []
        for i, component in enumerate(self.svd_.components_):
            order = np.argsort(np.abs(component))[-n_terms:][::-1]
            output.append({"component": i + 1,
                           "terms": [{"term": str(words[j]), "loading": float(component[j])} for j in order]})
        return output
