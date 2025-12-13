# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd


IndexKind = Literal["positional", "label"]


@dataclass(frozen=True)
class SplitIndices:
    """Container for train/test indices."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    index_kind: IndexKind = "positional"


def load_raw_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw tabular dataset from CSV or Parquet.

    Notes:
    - No preprocessing here (no filling NaNs, scaling, encoding, etc.).
    - Keep this function strictly for I/O.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        df = pd.read_csv(path)
    elif suffix in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file extension: {suffix}. Use .csv or .parquet"
        )

    if df.shape[0] == 0:
        raise ValueError("Loaded dataframe is empty.")
    return df


def load_splits(
    splits_dir: Union[str, Path],
    train_filename: str = "train_indices.npy",
    test_filename: str = "test_indices.npy",
    index_kind: IndexKind = "positional",
) -> SplitIndices:
    """
    Load train/test indices from .npy files.

    index_kind:
      - "positional": indices are integer positions (to be used with df.iloc)
      - "label": indices are labels (to be used with df.loc)
    """
    splits_dir = Path(splits_dir)
    train_path = splits_dir / train_filename
    test_path = splits_dir / test_filename

    if not train_path.exists():
        raise FileNotFoundError(f"Train indices file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test indices file not found: {test_path}")

    train_idx = np.load(train_path)
    test_idx = np.load(test_path)

    train_idx = _ensure_1d_int_array(train_idx, name="train_idx")
    test_idx = _ensure_1d_int_array(test_idx, name="test_idx")

    return SplitIndices(train_idx=train_idx, test_idx=test_idx, index_kind=index_kind)


def save_splits(
    splits_dir: Union[str, Path],
    split_indices: SplitIndices,
    train_filename: str = "train_indices.npy",
    test_filename: str = "test_indices.npy",
) -> None:
    """
    Save train/test indices to .npy files.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / train_filename, split_indices.train_idx)
    np.save(splits_dir / test_filename, split_indices.test_idx)


def validate_splits(
    df: pd.DataFrame,
    split_indices: SplitIndices,
    *,
    allow_partial: bool = True,
) -> None:
    """
    Validate that split indices are correct and non-overlapping.

    allow_partial:
      - True: train+test can be less than all rows (e.g., if you keep some rows unused)
      - False: train+test must cover all rows exactly once
    """
    train_idx = split_indices.train_idx
    test_idx = split_indices.test_idx

    # Check overlap
    inter = np.intersect1d(train_idx, test_idx)
    if inter.size > 0:
        raise ValueError(
            f"Train/Test indices overlap. Overlap size={inter.size}. "
            f"Example indices: {inter[:10].tolist()}"
        )

    # Check duplicates within each split
    if np.unique(train_idx).size != train_idx.size:
        raise ValueError("Train indices contain duplicates.")
    if np.unique(test_idx).size != test_idx.size:
        raise ValueError("Test indices contain duplicates.")

    # Range checks depend on index_kind
    if split_indices.index_kind == "positional":
        n = len(df)
        # Check for empty splits
        if train_idx.size == 0:
            if not allow_partial:
                raise ValueError("Train indices are empty, but allow_partial is False.")
        else:
            out_of_range_train = train_idx[(train_idx < 0) | (train_idx >= n)]
            if out_of_range_train.size > 0:
                raise ValueError(
                    f"Train indices out of range for df.iloc. "
                    f"Out-of-range values (up to 10 shown): {out_of_range_train[:10].tolist()}"
                )
        if test_idx.size == 0:
            if not allow_partial:
                raise ValueError("Test indices are empty, but allow_partial is False.")
        else:
            out_of_range_test = test_idx[(test_idx < 0) | (test_idx >= n)]
            if out_of_range_test.size > 0:
                raise ValueError(
                    f"Test indices out of range for df.iloc. "
                    f"Out-of-range values (up to 10 shown): {out_of_range_test[:10].tolist()}"
                )

        used = train_idx.size + test_idx.size
        if not allow_partial and used != n:
            raise ValueError(
                f"Splits do not cover all rows: used={used}, total={n}."
            )

    elif split_indices.index_kind == "label":
        # For label-based indexing, ensure indices exist in df.index
        df_index_values = np.asarray(df.index)
        missing_train = np.setdiff1d(train_idx, df_index_values)
        missing_test = np.setdiff1d(test_idx, df_index_values)
        if missing_train.size > 0:
            raise ValueError(
                f"Some train index labels are missing in df.index. "
                f"Example: {missing_train[:10].tolist()}"
            )
        if missing_test.size > 0:
            raise ValueError(
                f"Some test index labels are missing in df.index. "
                f"Example: {missing_test[:10].tolist()}"
            )

        used = train_idx.size + test_idx.size
        if not allow_partial and used != len(df_index_values):
            raise ValueError(
                f"Splits do not cover all rows: used={used}, total={len(df_index_values)}."
            )
    else:
        raise ValueError(f"Unknown index_kind: {split_indices.index_kind}")


def make_train_test(
    df: pd.DataFrame,
    target_col: str,
    split_indices: SplitIndices,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create X_train, y_train, X_test, y_test using provided split indices.

    IMPORTANT:
    - No feature processing here. Only slicing.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe columns.")

    validate_splits(df, split_indices, allow_partial=True)

    if split_indices.index_kind == "positional":
        train_df = df.iloc[split_indices.train_idx]
        test_df = df.iloc[split_indices.test_idx]
    else:  # "label"
        train_df = df.loc[split_indices.train_idx]
        test_df = df.loc[split_indices.test_idx]

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    return X_train, y_train, X_test, y_test


def load_dataset_and_splits(
    data_path: Union[str, Path],
    splits_dir: Union[str, Path],
    target_col: str,
    *,
    train_filename: str = "train_indices.npy",
    test_filename: str = "test_indices.npy",
    index_kind: IndexKind = "positional",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Convenience function: load raw data + splits and return train/test X,y.
    """
    df = load_raw_data(data_path)
    split_indices = load_splits(
        splits_dir,
        train_filename=train_filename,
        test_filename=test_filename,
        index_kind=index_kind,
    )
    return make_train_test(df, target_col=target_col, split_indices=split_indices)


def _ensure_1d_int_array(arr: np.ndarray, *, name: str) -> np.ndarray:
    """Ensure indices are a 1D numpy array of integers."""
    arr = np.asarray(arr)

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    # Many numpy loads come as int64 already; enforce integer dtype.
    if not np.issubdtype(arr.dtype, np.integer):
        # Try safe conversion if values look like ints
        if np.all(np.equal(arr, np.floor(arr))):
            arr = arr.astype(np.int64)
        else:
            raise TypeError(f"{name} must be integer indices. Got dtype={arr.dtype}")

    return arr.astype(np.int64)