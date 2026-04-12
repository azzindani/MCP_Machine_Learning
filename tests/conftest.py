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
