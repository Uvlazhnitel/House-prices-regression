from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold

from .data import load_dataset_and_splits
from .features import get_feature_groups, build_preprocessor, sanity_check_feature_groups
from .model import build_model, HGBParams
from .eval import (
    evaluate_cv,
    upsert_metrics_table,
    get_oof_predictions,
    save_residual_plots,
    save_top_errors,
    save_learning_curve_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate P2 regression model (HGB + preprocessing).")

    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to raw dataset file (CSV or Parquet), e.g. data/raw/train.csv")
    parser.add_argument("--splits-dir", type=str, required=True,
                        help="Directory with train/test indices .npy, e.g. data/splits/")
    parser.add_argument("--target-col", type=str, required=True,
                        help="Target column name, e.g. SalePrice")

    parser.add_argument("--index-kind", type=str, default="positional", choices=["positional", "label"],
                        help="How to interpret indices in splits: positional for df.iloc, label for df.loc")

    parser.add_argument("--target-transform", type=str, default="log1p", choices=["raw", "log1p"],
                        help="Target transform: raw or log1p (via TransformedTargetRegressor)")

    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of KFold splits")
    parser.add_argument("--main-metric", type=str, default="rmse", choices=["rmse", "mae", "r2"],
                        help="Main metric for CV summary/table")
    parser.add_argument("--secondary-metric", type=str, default="mae", choices=["rmse", "mae", "r2"],
                        help="Secondary metric for CV summary/table")

    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save reports")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Directory to save figures")

    parser.add_argument("--metrics-table", type=str, default="reports/metrics_models.csv",
                        help="CSV path for metrics table")
    parser.add_argument("--metrics-md", type=str, default="reports/metrics.md",
                        help="Markdown file to append training summary")

    parser.add_argument("--model-name", type=str, default="HGB_tuned",
                        help="Model name used in metrics table and logs")

    # Optional outputs
    parser.add_argument("--save-oof", action="store_true",
                        help="If set, compute OOF predictions and save residual plots + top errors")
    parser.add_argument("--save-learning-curve", action="store_true",
                        help="If set, compute and save learning curve plot")

    return parser.parse_args()


def build_cv(args: argparse.Namespace) -> KFold:
    return KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)


def default_top_error_cols(X: pd.DataFrame) -> list[str]:
    """
    Choose a few common columns for inspection if they exist in X.
    (This keeps the output useful without hardcoding dataset-specific columns.)
    """
    candidates = [
        "Neighborhood", "GrLivArea", "OverallQual", "YearBuilt",
        "TotalBsmtSF", "GarageArea", "LotArea"
    ]
    return [c for c in candidates if c in X.columns]


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def collect_env_metadata() -> Dict[str, Any]:
    """Collect lightweight metadata for reproducibility."""
    import sklearn  # local import for version
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    return meta


def append_metrics_md(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# Metrics log\n\n", encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        f.write("\n")


def main() -> None:
    args = parse_args()

    ensure_dirs(args.models_dir, args.reports_dir, args.figures_dir)

    # 1) Load data and splits -> X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = load_dataset_and_splits(
        data_path=args.data_path,
        splits_dir=args.splits_dir,
        target_col=args.target_col,
        index_kind=args.index_kind,  # "positional" by default
    )

    # 2) Build preprocessor
    groups = get_feature_groups(X_train, drop_cols=("Id",))
    sanity_check_feature_groups(X_train, groups)
    preprocessor = build_preprocessor(groups)

    # 3) Build model (HGB pipeline, optionally log1p target)
    hgb_params = HGBParams(
        max_iter=120,
        learning_rate=0.1254335218612733,
        max_depth=None,
        random_state=args.random_state,
        min_samples_leaf=11,
        max_leaf_nodes=23,
        early_stopping=True,
        validation_fraction=0.1,
    )

    model = build_model(
        preprocessor=preprocessor,
        target_transform=args.target_transform,  # "raw" or "log1p"
        hgb_params=hgb_params,
    )

    # 4) CV evaluation (same protocol as earlier sessions)
    cv = build_cv(args)
    summary = evaluate_cv(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        model_name=args.model_name,
        main_metric=args.main_metric,
        secondary_metric=args.secondary_metric,
        n_jobs=-1,
    )

    # 5) Save / update metrics table
    upsert_metrics_table(args.metrics_table, summary)

    # 6) Optional: OOF predictions + residual plots + top errors
    if args.save_oof:
        oof_pred = get_oof_predictions(model, X_train, y_train, cv=cv, n_jobs=-1)
        _ = save_residual_plots(
            y_true=y_train,
            y_pred=oof_pred,
            out_dir=args.figures_dir,
            prefix=f"{args.model_name}_oof",
        )
        top_errors_path = Path(args.reports_dir) / "top_errors.csv"
        _ = save_top_errors(
            X=X_train,
            y_true=y_train,
            y_pred=oof_pred,
            out_path=top_errors_path,
            top_n=50,
            extra_cols=default_top_error_cols(X_train),
        )

    # 7) Optional: learning curve
    if args.save_learning_curve:
        lc_path = Path(args.figures_dir) / "learning_curve.png"
        _ = save_learning_curve_plot(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            out_path=lc_path,
            scoring_metric=args.main_metric,
        )

    # 8) Fit final model on full training data and save artifacts
    model.fit(X_train, y_train)

    model_out_path = Path(args.models_dir) / "model.joblib"
    dump(model, model_out_path)

    meta = {
        **collect_env_metadata(),
        "data_path": str(Path(args.data_path)),
        "splits_dir": str(Path(args.splits_dir)),
        "target_col": args.target_col,
        "index_kind": args.index_kind,
        "target_transform": args.target_transform,
        "random_state": args.random_state,
        "cv_splits": args.cv_splits,
        "main_metric": args.main_metric,
        "secondary_metric": args.secondary_metric,
        "model_name": args.model_name,
        "feature_groups": {
            "numeric_cols": groups.numeric_cols,
            "categorical_cols": groups.categorical_cols,
        },
        "hgb_params": asdict(hgb_params),
        "cv_summary": {
            "main_metric": summary.main_metric,
            "main_mean": summary.main_mean,
            "main_std": summary.main_std,
            "secondary_metric": summary.secondary_metric,
            "secondary_mean": summary.secondary_mean,
            "secondary_std": summary.secondary_std,
            "n_splits": summary.n_splits,
        },
        "artifacts": {
            "model_joblib": str(model_out_path),
            "metrics_table": str(Path(args.metrics_table)),
        },
    }

    meta_path = Path(args.models_dir) / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 9) Append short log to metrics.md
    log_lines = [
        f"## Train run: {args.model_name}",
        f"- UTC time: {meta['timestamp_utc']}",
        f"- Target: `{args.target_col}` (transform: `{args.target_transform}`)",
        f"- CV: KFold(n_splits={args.cv_splits}, shuffle=True, random_state={args.random_state})",
        f"- Main metric: **{summary.main_metric}** = {summary.main_mean:.6f} ± {summary.main_std:.6f}",
    ]
    if (
        summary.secondary_metric is not None
        and summary.secondary_mean is not None
        and summary.secondary_std is not None
    ):
        log_lines.append(
            f"- Secondary metric: **{summary.secondary_metric}** = {summary.secondary_mean:.6f} ± {summary.secondary_std:.6f}"
        )
    log_lines.extend([
        f"- Saved model: `{model_out_path}`",
        f"- Saved meta: `{meta_path}`",
    ])
    log_text = "\n".join(log_lines)
    append_metrics_md(args.metrics_md, log_text)

    print("Done.")
    print(f"Metrics table updated: {args.metrics_table}")
    print(f"Model saved to: {model_out_path}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
