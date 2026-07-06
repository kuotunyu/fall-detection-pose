"""URFD 標註 CSV → ground-truth 跌倒事件區間。

URFD 的 ``urfall-cam0-falls.csv`` / ``urfall-cam0-adls.csv`` 無標頭,
欄位依序為:sequenceName, frameNumber, label, 之後為深度特徵欄(本專案不用)。
label 語意(官方):-1 = 未躺、0 = 跌落中(過渡姿態)、1 = 躺地。

GT 事件慣例(文獻對正類定義分歧,本專案明文寫死,README 同步聲明):

    GT 事件區間 = [第一個 label=0 的幀, 緊接其後連續 label=1 區段的最後一幀]

理由:告警依設計在「躺地持續」後觸發;若 GT 只含 label=0 的短暫過渡段,
設計正確的告警反而會被判為 miss。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_annotations(csv_path: str | Path) -> pd.DataFrame:
    """讀取 URFD 標註 CSV,回傳欄位 ``sequence / frame / label`` 的 DataFrame。"""
    df = pd.read_csv(csv_path, header=None)
    df = df.iloc[:, :3]
    df.columns = ["sequence", "frame", "label"]
    df["frame"] = df["frame"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def gt_interval(labels: pd.DataFrame) -> tuple[int, int] | None:
    """單一序列的 GT 事件區間(幀號);無跌倒標記(純 ADL)回傳 None。

    邊界處理:
    - 若 0 之後找不到任何 1(理論上不會發生):以連續 0 區段末幀收尾;
    - 若完全沒有 0 但有 1(不合規格的標註):以 1 區段起訖充當。
    """
    seq = labels.sort_values("frame")
    frames = seq["frame"].to_numpy()
    lab = seq["label"].to_numpy()

    zero_idx = [i for i, v in enumerate(lab) if v == 0]
    if not zero_idx:
        one_idx = [i for i, v in enumerate(lab) if v == 1]
        if not one_idx:
            return None
        start = one_idx[0]
        end = start
        while end + 1 < len(lab) and lab[end + 1] == 1:
            end += 1
        return int(frames[start]), int(frames[end])

    start = zero_idx[0]
    # 找 0 之後第一個 1,再走到該連續 1 區段結束
    i = start
    first_one = None
    for j in range(start, len(lab)):
        if lab[j] == 1:
            first_one = j
            break
    if first_one is None:
        end = start
        while end + 1 < len(lab) and lab[end + 1] == 0:
            end += 1
        return int(frames[start]), int(frames[end])

    end = first_one
    while end + 1 < len(lab) and lab[end + 1] == 1:
        end += 1
    return int(frames[start]), int(frames[end])


def load_gt_events(csv_path: str | Path) -> dict[str, tuple[int, int]]:
    """整份標註 CSV → {sequence 名: (起幀, 訖幀)};無事件的序列不在字典中。"""
    df = load_annotations(csv_path)
    out: dict[str, tuple[int, int]] = {}
    for name, grp in df.groupby("sequence", sort=True):
        interval = gt_interval(grp)
        if interval is not None:
            out[str(name)] = interval
    return out
