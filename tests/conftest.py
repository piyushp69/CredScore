"""Shared fixtures: a model bundle trained by the real pipeline CLI on synthetic data."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credscore import config
from credscore.pipeline.__main__ import main as pipeline_main
from credscore.service import ScoringService
from tests.synthetic import make_raw_tables


@pytest.fixture(scope="session")
def raw_tables():
    return make_raw_tables()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, raw_tables):
    path = tmp_path_factory.mktemp("data")
    for key, filename in config.RAW_TABLES.items():
        raw_tables[key].to_csv(path / filename, index=False)
    return path


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, data_dir):
    path = tmp_path_factory.mktemp("models")
    exit_code = pipeline_main([
        "all", "--data-dir", str(data_dir), "--model-dir", str(path),
        "--device", "cpu", "--max-rounds", "80", "--learning-rate", "0.15",
    ])
    assert exit_code == 0
    return path


@pytest.fixture(scope="session")
def service(model_dir) -> ScoringService:
    return ScoringService.load(model_dir)


@pytest.fixture(scope="session")
def client(model_dir):
    from backend.app import create_app

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CREDSCORE_MODEL_DIR", str(model_dir))
        with TestClient(create_app()) as test_client:
            yield test_client
