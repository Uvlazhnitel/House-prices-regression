# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class FeatureGroups:
    """Container for feature column groups."""
    numeric_cols: List[str]
    categorical_cols: List[str]


def get_feature_groups(
    X: pd.DataFrame,
    *,
    drop_cols: Optional[Sequence[str]] = ("Id",),
) -> FeatureGroups:
    """
    Identify numeric and categorical columns from a pandas DataFrame.

    Notes:
    - This function only inspects dtypes. It does NOT modify X.
    - Commonly, 'Id' is dropped from numeric columns to avoid leakage/meaningless signal.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.to_list()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()

    # Drop unwanted columns (e.g. Id) from both lists if present
    if drop_cols:
        drop_set = set(drop_cols)
        numeric_cols = [c for c in numeric_cols if c not in drop_set]
        categorical_cols = [c for c in categorical_cols if c not in drop_set]

    # Optional: ensure stable order for reproducibility
    numeric_cols = sorted(numeric_cols)
    categorical_cols = sorted(categorical_cols)

    return FeatureGroups(numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def build_preprocessor(
    feature_groups: FeatureGroups,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer with:
    - numeric: median imputation + standard scaling
    - categorical: most_frequent imputation + OrdinalEncoder with unknown mapped to -1

    This matches the preprocessing used in your notebook.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_groups.numeric_cols),
            ("cat", categorical_transformer, feature_groups.categorical_cols),
        ],
        remainder="drop",  # drop any columns not listed above
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_preprocessed_feature_names(
    preprocessor: ColumnTransformer,
) -> List[str]:
    """
    Return output feature names after preprocessing.

    Notes:
    - With OrdinalEncoder, categorical features keep one column per original categorical feature.
    - With StandardScaler, numeric features keep one column per original numeric feature.
    - Therefore the output names are usually like: 'num__GrLivArea', 'cat__Neighborhood', etc.
    """
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise TypeError(
            "Preprocessor does not support get_feature_names_out(). "
            "Use sklearn >= 1.0."
        )

    return preprocessor.get_feature_names_out().tolist()


def sanity_check_feature_groups(
    X: pd.DataFrame,
    feature_groups: FeatureGroups,
) -> None:
    """
    Basic checks to ensure selected columns exist and are disjoint.
    """
    cols = set(X.columns)
    missing_num = [c for c in feature_groups.numeric_cols if c not in cols]
    missing_cat = [c for c in feature_groups.categorical_cols if c not in cols]
    if missing_num:
        raise ValueError(f"Missing numeric columns in X: {missing_num[:10]}")
    if missing_cat:
        raise ValueError(f"Missing categorical columns in X: {missing_cat[:10]}")

    overlap = set(feature_groups.numeric_cols).intersection(feature_groups.categorical_cols)
    if overlap:
        raise ValueError(f"Columns overlap between numeric and categorical: {sorted(overlap)[:10]}")
