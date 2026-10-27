"""Load and validate the AI4I predictive maintenance dataset."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)

NUMERIC_COLUMNS = (
    "UDI",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)
EXPECTED_MACHINE_TYPES = frozenset({"L", "M", "H"})


def default_data_path() -> Path:
    """Return the configured CSV path or the repository's conventional path."""
    configured_path = os.environ.get("PREDICTIVE_MAINTENANCE_DATA")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path(__file__).resolve().parents[1] / "dataset" / "ai4i2020.csv"


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset and validate its required schema and value types.

    Missing values are retained for EDA and are reported by downstream
    preprocessing rather than silently filled during data loading.
    """
    csv_path = Path(path).expanduser() if path is not None else default_data_path()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download ai4i2020.csv and either "
            "place it in dataset/ai4i2020.csv or set "
            "PREDICTIVE_MAINTENANCE_DATA to its path."
        )

    try:
        dataframe = pd.read_csv(csv_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read dataset CSV at {csv_path}: {error}") from error

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    dataframe = dataframe.copy()
    for column in NUMERIC_COLUMNS:
        try:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column {column!r} contains non-numeric values."
            ) from error

    present_types = set(dataframe["Type"].dropna().astype(str).unique())
    unexpected_types = sorted(present_types - EXPECTED_MACHINE_TYPES)
    if unexpected_types:
        raise ValueError(
            "Unexpected machine Type values: " + ", ".join(unexpected_types)
        )

    observed_targets = set(dataframe["Machine failure"].dropna().unique())
    unexpected_targets = observed_targets - {0, 1}
    if unexpected_targets:
        raise ValueError(
            "Machine failure must contain only 0 or 1; found "
            + ", ".join(map(str, sorted(unexpected_targets)))
        )

    return dataframe
