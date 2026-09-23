"""Fetch the Home Credit competition files from Kaggle into the data directory.

Requires Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME /
KAGGLE_KEY) and acceptance of the competition rules on kaggle.com.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

EXTRA_FILES = ["HomeCredit_columns_description.csv"]


def missing_raw_files(data_dir: Path) -> list[str]:
    return [name for name in config.RAW_TABLES.values() if not (data_dir / name).exists()]


def download(data_dir: Path | None = None, force: bool = False) -> Path:
    data_dir = Path(data_dir or config.data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    if not force and not missing_raw_files(data_dir):
        log.info("Raw data already present in %s", data_dir)
        return data_dir

    import kagglehub

    log.info("Downloading '%s' from Kaggle...", config.KAGGLE_COMPETITION)
    source = Path(kagglehub.competition_download(config.KAGGLE_COMPETITION))
    for name in [*config.RAW_TABLES.values(), *EXTRA_FILES]:
        target = data_dir / name
        if (source / name).exists():
            shutil.copy2(source / name, target)
        elif (source / f"{name}.zip").exists():
            with zipfile.ZipFile(source / f"{name}.zip") as archive:
                archive.extract(name, data_dir)
        else:
            log.warning("%s not found in the Kaggle download at %s", name, source)

    still_missing = missing_raw_files(data_dir)
    if still_missing:
        raise FileNotFoundError(f"Download incomplete, missing: {still_missing}")
    log.info("Raw data ready in %s", data_dir)
    return data_dir
