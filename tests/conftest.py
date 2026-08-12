"""Test configuration and fixtures."""

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def classification_simple(tmp_path) -> Path:
    dst = tmp_path / "classification_simple.csv"
    shutil.copy(FIXTURES_DIR / "classification_simple.csv", dst)
    return dst


@pytest.fixture
def classification_messy(tmp_path) -> Path:
    dst = tmp_path / "classification_messy.csv"
    shutil.copy(FIXTURES_DIR / "classification_messy.csv", dst)
    return dst


@pytest.fixture
def regression_simple(tmp_path) -> Path:
    dst = tmp_path / "regression_simple.csv"
    shutil.copy(FIXTURES_DIR / "regression_simple.csv", dst)
    return dst


@pytest.fixture
def ad_data_real_sample_with_ctr(tmp_path) -> Path:
    """294-row real (not synthetic) stratified sample of Ad_Data.csv, with a
    derived ctr_pct column added exactly as it was during the 2026-08-12
    comprehensive real-world sweep (clicks/impressions*100 via data_basic
    column_math). Includes 4 genuine impressions=0/clicks>0 rows, so ctr_pct
    genuinely contains +inf — the real pattern that broke read_column_profile
    and every training tool before the _auto_preprocess fix."""
    import pandas as pd

    df = pd.read_csv(FIXTURES_DIR / "ad_data_real_sample.csv")
    df["ctr_pct"] = df["clicks"] / df["impressions"] * 100
    dst = tmp_path / "ad_data_real_sample_with_ctr.csv"
    df.to_csv(dst, index=False)
    return dst


@pytest.fixture
def regression_messy(tmp_path) -> Path:
    dst = tmp_path / "regression_messy.csv"
    shutil.copy(FIXTURES_DIR / "regression_messy.csv", dst)
    return dst


@pytest.fixture
def clustering_simple(tmp_path) -> Path:
    dst = tmp_path / "clustering_simple.csv"
    shutil.copy(FIXTURES_DIR / "clustering_simple.csv", dst)
    return dst


@pytest.fixture
def large_10k(tmp_path) -> Path:
    dst = tmp_path / "large_10k.csv"
    shutil.copy(FIXTURES_DIR / "large_10k.csv", dst)
    return dst


@pytest.fixture(autouse=True)
def constrained_mode_off(monkeypatch):
    """Default: run tests in standard (non-constrained) mode."""
    monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)


@pytest.fixture(autouse=True)
def isolate_output_dir(monkeypatch, tmp_path):
    """Redirect all get_output_dir() calls to tmp_path to avoid polluting ~/Downloads."""
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))


@pytest.fixture
def constrained_mode(monkeypatch):
    """Enable constrained mode for a specific test."""
    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
