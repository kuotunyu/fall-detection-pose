"""tune/test 切分:UR Fall Detection Dataset 70 支影片的分層抽樣。

切分比例在規劃階段已與使用者確認並固定:tune = 10 falls + 13 adls(約 1/3),
test = 20 falls + 27 adls(約 2/3、佔多數以求指標穩定)。fall/adl 各自獨立
洗牌(分層),避免其中一類意外集中在同一個 split。

seed 固定為 42 且結果直接存進版控(``eval/splits.yaml``,repo 根目錄),
不在 notebook 裡即時重新產生——切分只決定一次,之後每次執行都讀同一份
名單,避免任何環境差異導致 test split 悄悄漂移(那樣所有 test 數字都不可信)。
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from ..io.urfd import adl_sequences, fall_sequences

DEFAULT_SEED = 42
N_TUNE_FALLS = 10
N_TUNE_ADLS = 13


def generate_splits(
    seed: int = DEFAULT_SEED,
    n_tune_falls: int = N_TUNE_FALLS,
    n_tune_adls: int = N_TUNE_ADLS,
) -> dict:
    """分層洗牌切分:falls 與 adls 各自獨立洗牌,避免其中一類集中在同一 split。"""
    rng = random.Random(seed)
    falls = fall_sequences()
    adls = adl_sequences()
    rng.shuffle(falls)
    rng.shuffle(adls)
    return {
        "seed": seed,
        "tune": {
            "falls": sorted(falls[:n_tune_falls]),
            "adls": sorted(adls[:n_tune_adls]),
        },
        "test": {
            "falls": sorted(falls[n_tune_falls:]),
            "adls": sorted(adls[n_tune_adls:]),
        },
    }


def save_splits(splits: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(splits, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def load_splits(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_of(sequence: str, splits: dict) -> str:
    """回傳某序列屬於 ``"tune"`` 或 ``"test"``;不在名單中就拋例外,
    及早抓出序列名拼字錯誤或名單過期。"""
    for split_name in ("tune", "test"):
        group = splits[split_name]
        if sequence in group["falls"] or sequence in group["adls"]:
            return split_name
    raise KeyError(f"{sequence} 不在任何 split 名單中")
