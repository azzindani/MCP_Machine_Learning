"""Test configuration and fixtures.

All CSV fixtures are copied to a temp directory inside the user's home dir so
that resolve_path()'s security boundary check (home-dir enforcement) passes.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Use home-dir-based tmpdir so resolve_path() security check passes
_HOME = Path.home()


@pytest.fixture
def home_tmp():
    """A temporary directory inside the user's home directory."""
    with tempfile.TemporaryDirectory(dir=_HOME) as td:
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
