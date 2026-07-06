"""單幀幾何特徵的純函式測試。"""

import math

import numpy as np

from fall_detection.rules.features import compute_frame_geometry

from synthetic import LIE, SIT, STAND


def _geo(pose, L=100.0, conf=0.9, bbox=None, conf_min=0.35):
    kpts = pose * L + np.array([320.0, 240.0])
    if bbox is None:
        x1, y1 = kpts.min(axis=0) - 10
        x2, y2 = kpts.max(axis=0) + 10
        bbox = np.array([x1, y1, x2, y2])
    kconf = np.full(17, conf, dtype=np.float32)
    return compute_frame_geometry(kpts.reshape(-1), kconf, bbox, conf_min)


def test_upright_geometry():
    g = _geo(STAND)
    assert g.valid
    assert g.theta_deg < 5.0
    assert abs(g.torso_len - 100.0) < 2.0
    assert g.bbox_aspect < 1.0
    assert g.ankle_valid
    assert g.hip_ankle_gap > 150.0  # 1.8 L


def test_lying_geometry():
    g = _geo(LIE)
    assert g.valid
    assert g.theta_deg > 80.0
    assert g.bbox_aspect > 1.0
    assert g.hip_ankle_gap < 20.0  # 0.05 L


def test_sitting_geometry():
    g = _geo(SIT)
    assert g.valid
    assert 5.0 < g.theta_deg < 30.0
    assert g.bbox_aspect < 1.0


def test_inverted_clamps_to_90():
    # 髖高於肩(倒立):傾角 clamp 為 90
    pose = STAND.copy()
    pose[:, 1] = -pose[:, 1]
    g = _geo(pose)
    assert g.valid
    assert g.theta_deg == 90.0


def test_invisible_shoulders_invalid():
    kpts = (STAND * 100 + np.array([320.0, 240.0])).reshape(-1)
    kconf = np.full(17, 0.9, dtype=np.float32)
    kconf[5] = kconf[6] = 0.1  # 低於 kpt_conf_min
    g = compute_frame_geometry(kpts, kconf, np.array([0, 0, 100, 300]), 0.35)
    assert not g.valid


def test_sentinel_conf_invalid():
    # keypoints.conf is None → 整列 -1.0 哨兵 → 必然 invalid
    kpts = (STAND * 100 + np.array([320.0, 240.0])).reshape(-1)
    kconf = np.full(17, -1.0, dtype=np.float32)
    g = compute_frame_geometry(kpts, kconf, np.array([0, 0, 100, 300]), 0.35)
    assert not g.valid


def test_single_side_visible_still_valid():
    # 只剩單側肩/髖可見:以可見側代替中點,仍為 valid(Ambianic 同型設計)
    kpts = (STAND * 100 + np.array([320.0, 240.0])).reshape(-1)
    kconf = np.full(17, 0.9, dtype=np.float32)
    kconf[6] = kconf[12] = 0.0  # 右肩、右髖不可見
    g = compute_frame_geometry(kpts, kconf, np.array([0, 0, 100, 300]), 0.35)
    assert g.valid
    assert g.theta_deg < 10.0


def test_missing_ankles_flagged():
    kpts = (STAND * 100 + np.array([320.0, 240.0])).reshape(-1)
    kconf = np.full(17, 0.9, dtype=np.float32)
    kconf[15] = kconf[16] = 0.0
    g = compute_frame_geometry(kpts, kconf, np.array([0, 0, 100, 300]), 0.35)
    assert g.valid
    assert not g.ankle_valid
    assert math.isnan(g.hip_ankle_gap)


def test_degenerate_bbox_gives_nan_aspect():
    kpts = (STAND * 100 + np.array([320.0, 240.0])).reshape(-1)
    kconf = np.full(17, 0.9, dtype=np.float32)
    g = compute_frame_geometry(kpts, kconf, np.array([10, 10, 50, 10]), 0.35)
    assert g.valid
    assert math.isnan(g.bbox_aspect)
