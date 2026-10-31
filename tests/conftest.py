import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True  # keep the repo root free of __pycache__ when tests load dubber.py


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs GPU/models; deselect with -m 'not slow'")


@pytest.fixture(scope="session")
def clip_single():
    p = ROOT / "input" / "clip_single.mp4"
    if not p.exists():
        pytest.skip("run tests/make_test_clip.py --single first")
    return p
