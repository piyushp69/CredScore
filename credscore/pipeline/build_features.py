"""Raw Home Credit CSVs -> one-row-per-applicant base feature table (parquet)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .. import config
from ..features import (
    BUREAU_COLUMNS,
    ID_COLUMN,
    INSTALLMENT_COLUMNS,
    PREVIOUS_COLUMNS,
    TARGET_COLUMN,
    build_base_table,
)
from .download import missing_raw_files

log = logging.getLogger(__name__)


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, engine="pyarrow")
    except ImportError:  # pyarrow unavailable: slower, same result
        return pd.read_csv(path, usecols=usecols, low_memory=False)


def load_raw_tables(data_dir: Path, sample: int | None = None, seed: int = config.RANDOM_STATE) -> dict[str, pd.DataFrame]:
    missing = missing_raw_files(data_dir)
    if missing:
        raise FileNotFoundError(
            f"Missing raw files in {data_dir}: {missing}. Run `python -m credscore.pipeline download`."
        )
    files = config.RAW_TABLES
    application = _read_csv(data_dir / files["application"])
    if sample:
        application = application.sample(n=min(sample, len(application)), random_state=seed)
    ids = set(application[ID_COLUMN])

    tables = {"application": application}
    for key, columns in (("bureau", BUREAU_COLUMNS), ("previous", PREVIOUS_COLUMNS), ("installments", INSTALLMENT_COLUMNS)):
        table = _read_csv(data_dir / files[key], usecols=columns)
        tables[key] = table[table[ID_COLUMN].isin(ids)] if sample else table
    for key, table in tables.items():
        log.info("  %-13s %10s rows x %3d columns", key, f"{len(table):,}", table.shape[1])
    return tables


def build_features(data_dir: Path | None = None, sample: int | None = None) -> Path:
    data_dir = Path(data_dir or config.data_dir())
    start = time.perf_counter()
    log.info("Loading raw tables from %s", data_dir)
    tables = load_raw_tables(data_dir, sample)

    log.info("Aggregating side tables and building the base feature table...")
    base = build_base_table(tables["application"], tables["bureau"], tables["previous"], tables["installments"])

    output = data_dir / config.FEATURES_FILE
    base.to_parquet(output, index=False)
    log.info(
        "Wrote %s: %s applicants, %d columns, default rate %.2f%% (%.0fs)",
        output.name, f"{len(base):,}", base.shape[1], 100 * base[TARGET_COLUMN].mean(), time.perf_counter() - start,
    )
    return output


def load_features(data_dir: Path | None = None) -> pd.DataFrame:
    path = Path(data_dir or config.data_dir()) / config.FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m credscore.pipeline features` first.")
    return pd.read_parquet(path)
