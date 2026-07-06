"""合成關鍵點軌跡產生器:生成與 keypoint cache 同格式的 DataFrame。

不碰模型、不碰影片,專供規則引擎單元測試。骨架為簡化模板:
只有肩(5,6)、髖(11,12)、踝(15,16)承載幾何意義,其餘關鍵點
擺在合理位置,只影響 bbox 外形。

座標系:像素、y 向下;模板以「髖中點」為原點、以軀幹長 L 為單位。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fall_detection.io.cache import CACHE_COLUMNS, N_KPTS

# --- 17 點姿態模板(L 單位,髖中點為原點) --------------------------------

STAND = np.array(
    [
        (0.00, -1.25),  # 0 nose
        (-0.04, -1.30), (0.04, -1.30),  # eyes
        (-0.09, -1.27), (0.09, -1.27),  # ears
        (-0.18, -1.00), (0.18, -1.00),  # shoulders
        (-0.25, -0.55), (0.25, -0.55),  # elbows
        (-0.28, -0.15), (0.28, -0.15),  # wrists
        (-0.12, 0.00), (0.12, 0.00),    # hips
        (-0.11, 0.95), (0.11, 0.95),    # knees
        (-0.10, 1.80), (0.10, 1.80),    # ankles
    ]
)

SIT = np.array(
    [
        (0.30, -1.22),
        (0.26, -1.27), (0.34, -1.27),
        (0.22, -1.24), (0.38, -1.24),
        (0.08, -0.97), (0.44, -0.97),   # 肩中點 (0.26,-0.97) → 軀幹傾角 ≈ 15°
        (0.20, -0.50), (0.40, -0.50),
        (0.25, -0.10), (0.45, -0.10),
        (-0.12, 0.00), (0.12, 0.00),
        (0.45, 0.10), (0.55, 0.10),
        (0.30, 0.90), (0.40, 0.90),     # 踝中點 y=0.9 → h_hip ≈ 0.9
    ]
)

LIE = np.array(
    [
        (-1.30, -0.02),
        (-1.34, -0.06), (-1.34, 0.02),
        (-1.30, -0.08), (-1.30, 0.04),
        (-1.00, -0.08), (-1.00, 0.02),  # 肩中點 (-1.0,-0.03) → 傾角 ≈ 88°
        (-0.70, 0.08), (-0.70, 0.14),
        (-0.40, 0.10), (-0.40, 0.16),
        (-0.05, -0.04), (0.05, 0.04),
        (0.90, 0.00), (0.90, 0.08),
        (1.80, 0.02), (1.80, 0.08),     # 踝中點 y=0.05 → h_hip ≈ 0.05
    ]
)

POSES = {"stand": STAND, "walk": STAND, "sit": SIT, "lie": LIE}

# 髖中點離地高度(L 單位):決定跌倒/坐下時的垂直位移量
HIP_HEIGHT = {"stand": 1.80, "walk": 1.80, "sit": 0.90, "lie": 0.05}

BBOX_MARGIN = 0.15  # bbox 相對關鍵點外擴(L 單位)


def make_trajectory(
    segments: list[tuple[str, float]],
    fps: float = 30.0,
    L: float = 100.0,
    ground_y: float = 400.0,
    x0: float = 320.0,
    track_id: int = 1,
    noise: float = 0.5,
    seed: int = 0,
    walk_speed: float = 30.0,
) -> pd.DataFrame:
    """依分段描述生成 cache 格式的軌跡。

    Args:
        segments: ``[("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)]``;
            ``"to:X"`` 表示由目前姿態線性過渡到 X,其餘為維持該姿態。
            第一段不可為過渡段。
        walk_speed: ``walk`` 段的水平速度(px/s)。
    """
    rng = np.random.default_rng(seed)
    rows = []
    frame = 0
    x_off = 0.0
    cur_pose = segments[0][0].split(":")[-1]
    if segments[0][0].startswith("to:"):
        raise ValueError("第一段必須是維持姿態,不可為過渡段")

    for kind, dur in segments:
        n = max(1, round(dur * fps))
        is_trans = kind.startswith("to:")
        target = kind.split(":", 1)[1] if is_trans else kind
        p_from = POSES[cur_pose]
        p_to = POSES[target]
        h_from = HIP_HEIGHT[cur_pose]
        h_to = HIP_HEIGHT[target]
        for i in range(n):
            alpha = (i + 1) / n if is_trans else 1.0
            pose = p_from + (p_to - p_from) * alpha if is_trans else p_to
            hip_h = h_from + (h_to - h_from) * alpha if is_trans else h_to
            if kind == "walk":
                x_off += walk_speed / fps
            hip_xy = np.array([x0 + x_off, ground_y - hip_h * L])
            kpts = hip_xy + pose * L
            if noise > 0:
                kpts = kpts + rng.normal(0.0, noise, size=kpts.shape)
            x1, y1 = kpts.min(axis=0) - BBOX_MARGIN * L
            x2, y2 = kpts.max(axis=0) + BBOX_MARGIN * L
            rows.append(
                {
                    "frame_idx": np.int32(frame),
                    "t_ms": frame / fps * 1000.0,
                    "track_id": np.int32(track_id),
                    "bbox_x1": np.float32(x1),
                    "bbox_y1": np.float32(y1),
                    "bbox_x2": np.float32(x2),
                    "bbox_y2": np.float32(y2),
                    "bbox_conf": np.float32(0.9),
                    "kpts_xy": kpts.astype(np.float32).reshape(-1),
                    "kpts_conf": np.full(N_KPTS, 0.9, dtype=np.float32),
                }
            )
            frame += 1
        cur_pose = target
    return pd.DataFrame(rows, columns=CACHE_COLUMNS)


# --- 軌跡變換工具 --------------------------------------------------------


def drop_keypoints(df: pd.DataFrame, t0_s: float, t1_s: float, sentinel: float = 0.0) -> pd.DataFrame:
    """把 [t0, t1) 時間範圍內的 kpts_conf 全部設為 sentinel(模擬 keypoint dropout;
    sentinel=-1.0 可模擬 ultralytics ``keypoints.conf is None`` 的哨兵列)。"""
    df = df.copy()
    mask = (df["t_ms"] / 1000.0 >= t0_s) & (df["t_ms"] / 1000.0 < t1_s)
    df["kpts_conf"] = [
        np.full(N_KPTS, sentinel, dtype=np.float32) if m else c
        for m, c in zip(mask, df["kpts_conf"])
    ]
    return df


def switch_track(df: pd.DataFrame, t_switch_s: float, new_id: int) -> pd.DataFrame:
    """t ≥ t_switch 的列改用新 track id(模擬 ByteTrack 斷 id)。"""
    df = df.copy()
    mask = df["t_ms"] / 1000.0 >= t_switch_s
    df.loc[mask, "track_id"] = np.int32(new_id)
    return df


def scale_coords(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """所有像素座標乘上 factor(檢驗尺度不變性;factor 取 2 的冪可保浮點精確)。"""
    df = df.copy()
    for col in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        df[col] = (df[col] * factor).astype(np.float32)
    df["kpts_xy"] = [(k * factor).astype(np.float32) for k in df["kpts_xy"]]
    return df
