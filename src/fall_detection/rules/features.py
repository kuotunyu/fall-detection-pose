"""單幀幾何特徵(純函式:無狀態、無 I/O)。

座標系為影像像素、y 向下。只有置信度 ≥ ``kpt_conf_min`` 的關鍵點視為可見;
肩或髖完全不可見時整幀標記 invalid(軀幹是所有特徵的基準,不硬猜)。
時間域的平滑、正規化與速度計算不在此處——見 ``engine``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# COCO 17-keypoint 索引(ultralytics pose 模型輸出順序)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16

_EPS = 1e-6


@dataclass
class FrameGeometry:
    """一幀、一個人的原始幾何量(像素單位,未平滑、未正規化)。"""

    valid: bool
    theta_deg: float = float("nan")  # 軀幹傾角:0=直立、90=橫臥(頭低於髖時 clamp 90)
    torso_len: float = float("nan")  # 肩中點-髖中點距離(像素);引擎以其滑動中位數 L̃ 作尺度
    bbox_aspect: float = float("nan")  # bbox 寬/高
    hip_y: float = float("nan")  # 髖中點 y(像素)
    ankle_valid: bool = False
    hip_ankle_gap: float = float("nan")  # 踝中點 y − 髖中點 y(>0 = 踝在髖下方)


def _visible_mid(
    kpts_xy: np.ndarray, kpts_conf: np.ndarray, idxs: tuple[int, int], conf_min: float
) -> np.ndarray | None:
    # NaN 座標即使 conf 高也視為不可見:NaN 一旦進入中位數/差分緩衝會汙染整條管線
    pts = [
        kpts_xy[i]
        for i in idxs
        if kpts_conf[i] >= conf_min and np.all(np.isfinite(kpts_xy[i]))
    ]
    if not pts:
        return None
    return np.mean(np.asarray(pts, dtype=np.float64), axis=0)


def compute_frame_geometry(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    bbox: np.ndarray,
    kpt_conf_min: float,
) -> FrameGeometry:
    """由 17 個關鍵點與 bbox 計算單幀幾何特徵。

    Args:
        kpts_xy: 形狀 (17, 2) 或展平 (34,) 的像素座標。
        kpts_conf: 形狀 (17,) 的置信度;整列 -1.0 為「模型未輸出 conf」哨兵,
            必然低於 ``kpt_conf_min``,自動導致 invalid。
        bbox: (x1, y1, x2, y2)。
        kpt_conf_min: 關鍵點可見門檻。
    """
    kpts_xy = np.asarray(kpts_xy, dtype=np.float64).reshape(-1, 2)
    kpts_conf = np.asarray(kpts_conf, dtype=np.float64).reshape(-1)
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)

    shoulder = _visible_mid(kpts_xy, kpts_conf, (L_SHOULDER, R_SHOULDER), kpt_conf_min)
    hip = _visible_mid(kpts_xy, kpts_conf, (L_HIP, R_HIP), kpt_conf_min)
    if shoulder is None or hip is None:
        return FrameGeometry(valid=False)

    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    torso_len = math.hypot(dx, dy)
    if not math.isfinite(torso_len) or torso_len < _EPS:
        return FrameGeometry(valid=False)

    if dy < 0:
        # 髖高於肩(倒立/翻滾):傾角語意上已是「非直立」,clamp 為 90°
        theta = 90.0
    else:
        theta = math.degrees(math.atan2(abs(dx), dy))

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    aspect = float(w / h) if (math.isfinite(w) and h > _EPS) else float("nan")

    ankle = _visible_mid(kpts_xy, kpts_conf, (L_ANKLE, R_ANKLE), kpt_conf_min)
    ankle_valid = ankle is not None
    gap = float(ankle[1] - hip[1]) if ankle_valid else float("nan")

    return FrameGeometry(
        valid=True,
        theta_deg=float(theta),
        torso_len=float(torso_len),
        bbox_aspect=aspect,
        hip_y=float(hip[1]),
        ankle_valid=ankle_valid,
        hip_ankle_gap=gap,
    )
