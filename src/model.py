# src/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


TargetTransform = Literal["raw", "log1p"]


@dataclass(frozen=True)
class HGBParams:
    """Hyperparameters for HistGradientBoostingRegressor."""
    max_iter: int = 120
    learning_rate: float = 0.1254335218612733
    max_depth: Optional[int] = None
    random_state: int = 42
    min_samples_leaf: int = 11
    max_leaf_nodes: int = 23
    early_stopping: bool = True
    validation_fraction: float = 0.1


def build_hgb_regressor(params: HGBParams) -> HistGradientBoostingRegressor:
    """
    Build HistGradientBoostingRegressor with the tuned hyperparameters.
    """
    return HistGradientBoostingRegressor(
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        random_state=params.random_state,
        min_samples_leaf=params.min_samples_leaf,
        max_leaf_nodes=params.max_leaf_nodes,
        early_stopping=params.early_stopping,
        validation_fraction=params.validation_fraction,
    )


def build_hgb_pipeline(
    preprocessor: ColumnTransformer,
    params: HGBParams,
) -> Pipeline:
    """
    Build a Pipeline: preprocessor -> HistGradientBoostingRegressor.
    """
    regressor = build_hgb_regressor(params)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    return model


def build_model(
    preprocessor: ColumnTransformer,
    *,
    target_transform: TargetTransform = "raw",
    hgb_params: Optional[HGBParams] = None,
) -> Union[Pipeline, TransformedTargetRegressor]:
    """
    Build the final model. If target_transform == "log1p", wraps the pipeline
    into TransformedTargetRegressor with log1p/expm1.

    Returns:
      - Pipeline if target_transform == "raw"
      - TransformedTargetRegressor if target_transform == "log1p"
    """
    if hgb_params is None:
        hgb_params = HGBParams()

    base_pipeline = build_hgb_pipeline(preprocessor, hgb_params)

    if target_transform == "raw":
        return base_pipeline

    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=base_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    raise ValueError(f"Unknown target_transform: {target_transform}. Use 'raw' or 'log1p'.")
