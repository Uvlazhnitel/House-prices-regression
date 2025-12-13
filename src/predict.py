# src/predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict with a trained regression model (joblib). CSV in -> CSV out."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to trained model artifact (joblib), e.g. models/model.joblib",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV with features, e.g. data/raw/house_prices_test.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Path to save predictions CSV, e.g. reports/predictions.csv",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="Id",
        help="Optional ID column name to carry over to output (default: Id).",
    )
    parser.add_argument(
        "--drop-target-col",
        type=str,
        default="SalePrice",
        help="If present in input, this column will be dropped (default: SalePrice).",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="models/model_meta.json",
        help="Optional path to model metadata (json). Used only for basic sanity checks/logging.",
    )
    parser.add_argument(
        "--strict-columns",
        action="store_true",
        help="If set, enforce that input columns match training columns saved in meta (if available).",
    )
    return parser.parse_args()


def load_meta(meta_path: str) -> Optional[dict]:
    p = Path(meta_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def maybe_check_columns(df: pd.DataFrame, meta: Optional[dict], strict: bool) -> None:
    """
    Optionally check that input contains all columns used in training.
    We only do a light check since preprocessing handles missing categories, etc.
    """
    if meta is None:
        return

    fg = meta.get("feature_groups", {})
    train_cols = set(fg.get("numeric_cols", [])) | set(fg.get("categorical_cols", []))
    if not train_cols:
        return

    input_cols = set(df.columns)
    missing = sorted(list(train_cols - input_cols))
    extra = sorted(list(input_cols - train_cols))

    if missing:
        msg = (
            f"Input is missing {len(missing)} training columns. "
            f"Example: {missing[:10]}"
        )
        raise ValueError(msg)

    if strict and extra:
        # If strict, do not allow extra columns (except Id/target which may be handled elsewhere)
        msg = (
            f"Input has {len(extra)} extra columns not seen in training. "
            f"Example: {extra[:10]}"
        )
        raise ValueError(msg)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load(model_path)

    # Load meta (optional)
    meta = load_meta(args.meta_path)

    # Load input data
    df = pd.read_csv(input_path)
    if df.shape[0] == 0:
        raise ValueError("Input CSV is empty.")

    # Carry over Id if present
    id_series = df[args.id_col] if args.id_col in df.columns else None

    # Drop target if present (avoid leakage / accidental wrong usage)
    if args.drop_target_col in df.columns:
        df = df.drop(columns=[args.drop_target_col])

    # Optional column checks (light)
    maybe_check_columns(df, meta, strict=args.strict_columns)

    # Predict
    preds = model.predict(df)
    preds = np.asarray(preds, dtype=float)

    # Build output
    out_df = pd.DataFrame({"prediction": preds})
    if id_series is not None:
        out_df.insert(0, args.id_col, id_series.values)

    out_df.to_csv(out_path, index=False)

    print("Done.")
    print(f"Input:  {input_path} ({len(df)} rows)")
    print(f"Model:  {model_path}")
    if meta is not None:
        print(f"Meta:   {args.meta_path}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
