# src/eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, cross_val_predict, learning_curve
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import matplotlib.pyplot as plt


MetricName = Literal["rmse", "mae", "r2"]


@dataclass(frozen=True)
class CVSummary:
    """Cross-validation summary for one model."""
    model_name: str
    main_metric: MetricName
    main_mean: float
    main_std: float
    secondary_metric: Optional[MetricName] = None
    secondary_mean: Optional[float] = None
    secondary_std: Optional[float] = None
    n_splits: Optional[int] = None


def get_sklearn_scoring(metric: MetricName) -> str:
    """
    Map our metric name to sklearn scoring string.

    Notes:
    - For errors, sklearn uses "neg_*" scorers (higher is better).
    - We'll convert them back to positive errors later.
    """
    if metric == "rmse":
        return "neg_root_mean_squared_error"
    if metric == "mae":
        return "neg_mean_absolute_error"
    if metric == "r2":
        return "r2"
    raise ValueError(f"Unknown metric: {metric}")


def _to_positive_metric(metric: MetricName, scores: np.ndarray) -> np.ndarray:
    """
    Convert sklearn scores to human-readable metrics.
    For 'neg_*' scorers, flip sign to get positive error values.
    For r2, keep as is.
    """
    scores = np.asarray(scores, dtype=float)
    if metric in ("rmse", "mae"):
        return -scores
    return scores


def evaluate_cv(
    estimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv,
    *,
    model_name: str,
    main_metric: MetricName = "rmse",
    secondary_metric: Optional[MetricName] = "mae",
    n_jobs: int = -1,
) -> CVSummary:
    """
    Evaluate estimator with cross-validation and return mean/std for main + optional secondary metric.

    IMPORTANT:
    - If estimator is TransformedTargetRegressor (log1p), predict() returns values in original scale,
      so RMSE/MAE computed by sklearn scorers are in original scale too.
    """
    scoring: Dict[str, str] = {"main": get_sklearn_scoring(main_metric)}
    if secondary_metric is not None:
        scoring["secondary"] = get_sklearn_scoring(secondary_metric)

    cv_res = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False,
    )

    main_scores = _to_positive_metric(main_metric, cv_res["test_main"])
    main_mean = float(np.mean(main_scores))
    main_std = float(np.std(main_scores))

    sec_mean = sec_std = None
    if secondary_metric is not None:
        sec_scores = _to_positive_metric(secondary_metric, cv_res["test_secondary"])
        sec_mean = float(np.mean(sec_scores))
        sec_std = float(np.std(sec_scores))

    # Try to infer n_splits from CV object (best effort)
    n_splits = None
    if hasattr(cv, "get_n_splits"):
        try:
            n_splits = int(cv.get_n_splits(X, y))
        except Exception:
            n_splits = None

    return CVSummary(
        model_name=model_name,
        main_metric=main_metric,
        main_mean=main_mean,
        main_std=main_std,
        secondary_metric=secondary_metric,
        secondary_mean=sec_mean,
        secondary_std=sec_std,
        n_splits=n_splits,
    )


def get_oof_predictions(
    estimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv,
    *,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Get out-of-fold predictions for error analysis.

    Notes:
    - Uses cross_val_predict with method='predict'.
    - For TransformedTargetRegressor, predictions are in original scale.
    """
    y_pred = cross_val_predict(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        method="predict",
    )
    return np.asarray(y_pred, dtype=float)


def evaluate_on_data(
    estimator,
    X: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate fitted estimator on a dataset (no CV). Returns rmse, mae, r2.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(estimator.predict(X), dtype=float)

    rmse = float(mean_squared_error(y_true_arr, y_pred, squared=False))
    mae = float(mean_absolute_error(y_true_arr, y_pred))
    r2 = float(r2_score(y_true_arr, y_pred))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def upsert_metrics_table(
    table_path: Union[str, Path],
    summary: CVSummary,
) -> pd.DataFrame:
    """
    Insert or update a row in a metrics table CSV.

    Output columns:
      - model
      - main_metric, main_mean, main_std
      - secondary_metric, secondary_mean, secondary_std
      - n_splits
    """
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model": summary.model_name,
        "main_metric": summary.main_metric,
        "main_mean": summary.main_mean,
        "main_std": summary.main_std,
        "secondary_metric": summary.secondary_metric,
        "secondary_mean": summary.secondary_mean,
        "secondary_std": summary.secondary_std,
        "n_splits": summary.n_splits,
    }

    if table_path.exists():
        df = pd.read_csv(table_path)
        if "model" not in df.columns:
            raise ValueError(f"'model' column is missing in {table_path}")
        df = df[df["model"] != summary.model_name]  # drop old row if exists
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    # Sort for readability
    df = df.sort_values(by=["main_mean", "model"], ascending=[True, True]).reset_index(drop=True)
    df.to_csv(table_path, index=False)
    return df


def save_residual_plots(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    out_dir: Union[str, Path],
    *,
    prefix: str = "residuals",
    bins: int = 40,
) -> Tuple[Path, Path]:
    """
    Save two plots:
      1) histogram of residuals
      2) scatter: y_pred vs residual

    Returns paths to saved images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr

    # 1) Histogram
    hist_path = out_dir / f"{prefix}_hist.png"
    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.title("Residuals histogram (y_true - y_pred)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # 2) Scatter y_pred vs residual
    scatter_path = out_dir / f"{prefix}_pred_vs_resid.png"
    plt.figure()
    plt.scatter(y_pred_arr, residuals, s=10, alpha=0.6)
    plt.axhline(0.0, linewidth=1.0)
    plt.title("Predicted value vs residual")
    plt.xlabel("y_pred")
    plt.ylabel("residual = y_true - y_pred")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    return hist_path, scatter_path


def save_top_errors(
    X: pd.DataFrame,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    out_path: Union[str, Path],
    *,
    top_n: int = 50,
    extra_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Save a CSV with top absolute errors.

    Includes:
      - y_true, y_pred, residual, abs_error
      - optionally additional columns from X for inspection (e.g., Neighborhood, GrLivArea, OverallQual)

    Returns the dataframe saved.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residual = y_true_arr - y_pred_arr
    abs_error = np.abs(residual)

    df = pd.DataFrame(
        {
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "residual": residual,
            "abs_error": abs_error,
        }
    )

    if extra_cols:
        cols = [c for c in extra_cols if c in X.columns]
        for c in cols:
            df[c] = X[c].values

    df = df.sort_values("abs_error", ascending=False).head(int(top_n)).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def save_learning_curve_plot(
    estimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv,
    out_path: Union[str, Path],
    *,
    scoring_metric: MetricName = "rmse",
    train_sizes: Optional[Sequence[float]] = None,
    n_jobs: int = -1,
) -> Path:
    """
    Build and save a learning curve plot: training size vs train/validation score.

    Notes:
    - Uses sklearn.model_selection.learning_curve
    - For error metrics (rmse/mae) we convert from neg scores to positive errors for readability.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    scoring = get_sklearn_scoring(scoring_metric)

    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=42,
    )

    # Convert to positive errors if needed
    train_scores = _to_positive_metric(scoring_metric, train_scores)
    val_scores = _to_positive_metric(scoring_metric, val_scores)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure()
    plt.plot(sizes, train_mean, marker="o", label="Train")
    plt.plot(sizes, val_mean, marker="o", label="Validation")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.title("Learning curve")
    plt.xlabel("Training set size")
    ylabel = f"{scoring_metric.upper()} (lower is better)" if scoring_metric in ("rmse", "mae") else "R2 (higher is better)"
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path
