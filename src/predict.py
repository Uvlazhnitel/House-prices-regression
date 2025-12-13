# src/predict.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from joblib import load


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using a saved model (joblib).")

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to saved model.joblib",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input data file (.csv or .parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output predictions CSV. If not set, saved to predictions/predictions.csv",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="SalePrice",
        help="Target column name to drop if present in input",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="Id",
        help="ID column name to keep in the output if present",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="prediction",
        help="Prediction column name in the output file",
    )

    return parser.parse_args()


def load_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load a table from CSV or Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported input extension: {suffix}. Use .csv or .parquet")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model (Pipeline or TransformedTargetRegressor)
    model = load(model_path)

    # Load input data
    df = load_table(args.input_path)
    if df.shape[0] == 0:
        raise ValueError("Input dataframe is empty.")

    # Keep Id if present for nicer output (Kaggle-like submission)
    id_series: Optional[pd.Series] = df[args.id_col] if args.id_col in df.columns else None

    # Drop target if present (avoid leakage and shape mismatch)
    if args.target_col in df.columns:
        logging.info(f"Target column '{args.target_col}' found in input data and will be dropped to avoid leakage.")
    X = df.drop(columns=[args.target_col], errors="ignore")

    # Predict
    preds = model.predict(X)

    # Build output dataframe
    out_df = pd.DataFrame({args.pred_col: preds})
    if id_series is not None:
        out_df.insert(0, args.id_col, id_series.reset_index(drop=True))

    # Determine output path
    if args.output_path is None:
        out_path = Path("predictions") / "predictions.csv"
    else:
        out_path = Path(args.output_path)

    ensure_parent_dir(out_path)
    out_df.to_csv(out_path, index=False)

    print("Done.")
    print(f"Input:  {Path(args.input_path)}")
    print(f"Model:  {model_path}")
    print(f"Output: {out_path}")
    print(f"Rows:   {len(out_df)}")


if __name__ == "__main__":
    main()
