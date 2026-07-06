"""測試共用 fixtures:一律使用 repo 根目錄的真實 config.yaml,
確保測試驗證的是實際出貨的預設閾值,而不是另一套測試專用參數。"""

from pathlib import Path

import pytest

from fall_detection.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(REPO_ROOT / "config.yaml")
