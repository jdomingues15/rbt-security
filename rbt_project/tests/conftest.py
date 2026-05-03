"""
tests/conftest.py
Shared pytest configuration and fixtures.
"""
import pytest
import requests

BASE = "http://localhost:8000"

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ml: marks tests that require the ML model to be loaded"
    )

@pytest.fixture(scope="session", autouse=True)
def check_api_running():
    """Fail fast if the API is not running."""
    try:
        r = requests.get(f"{BASE}/", timeout=5)
        r.raise_for_status()
    except Exception:
        pytest.exit(
            "❌ API not reachable at http://localhost:8000\n"
            "   Run: docker compose up -d  before running tests",
            returncode=1
        )
