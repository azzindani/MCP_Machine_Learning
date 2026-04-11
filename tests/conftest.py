"""Test configuration and fixtures."""

import shutil
import tempfile
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def home_tmp():
    """A temporary directory for test files and outputs."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def classification_simple(home_tmp):
    dst = home_tmp / "classification_simple.csv"
    shutil.copy(FIXTURES_DIR / "classification_simple.csv", dst)
    return str(dst)


@pytest.fixture
def classification_messy(home_tmp):
    dst = home_tmp / "classification_messy.csv"
    shutil.copy(FIXTURES_DIR / "classification_messy.csv", dst)
    return str(dst)


@pytest.fixture
def regression_simple(home_tmp):
    dst = home_tmp / "regression_simple.csv"
    shutil.copy(FIXTURES_DIR / "regression_simple.csv", dst)
    return str(dst)


@pytest.fixture
def regression_messy(home_tmp):
    dst = home_tmp / "regression_messy.csv"
    shutil.copy(FIXTURES_DIR / "regression_messy.csv", dst)
    return str(dst)


@pytest.fixture
def clustering_simple(home_tmp):
    dst = home_tmp / "clustering_simple.csv"
    shutil.copy(FIXTURES_DIR / "clustering_simple.csv", dst)
    return str(dst)


@pytest.fixture
def large_10k(home_tmp):
    dst = home_tmp / "large_10k.csv"
    shutil.copy(FIXTURES_DIR / "large_10k.csv", dst)
    return str(dst)


@pytest.fixture(autouse=True)
def constrained_mode_off(monkeypatch):
    """Default: run tests in standard (non-constrained) mode."""
    monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)


@pytest.fixture
def constrained_mode(monkeypatch):
    """Enable constrained mode for a specific test."""
    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
