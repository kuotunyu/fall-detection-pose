"""引擎端到端測試:合成軌跡 → run_engine → 事件斷言。

全部使用 repo 根目錄 config.yaml 的真實預設閾值。
"""

import numpy as np
import pandas as pd

from fall_detection.io.cache import CACHE_COLUMNS, N_KPTS
from fall_detection.rules import run_engine

from synthetic import (
    drop_keypoints,
    make_trajectory,
    scale_coords,
    switch_track,
)

FPS = 30.0


def test_no_event_walking(cfg):
    df = make_trajectory([("walk", 10.0)], fps=FPS, noise=0.8)
    events, _ = run_engine(df, FPS, cfg)
    assert events == []


def test_textbook_fall_exactly_one_event(cfg):
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS)
    events, _ = run_engine(df, FPS, cfg)
    assert len(events) == 1
    ev = events[0]
    assert abs(ev.start_time_s - 2.0) <= 0.3  # 起點落在跌落段 ±0.3s
    assert ev.end_time_s >= 5.4  # 影片在躺地中結束:事件收在結尾
    assert ev.track_ids == [1]
    assert ev.peak_features["max_v_torso_per_s"] > cfg.rules.v_fall_enter


def test_fast_sit_rejected_by_timeout(cfg):
    df = make_trajectory([("stand", 2.0), ("to:sit", 0.4), ("sit", 2.5)], fps=FPS)
    events, debug = run_engine(df, FPS, cfg, collect_debug=True)
    assert events == []
    # 驗證確實觸發過 FALLING(速度夠快)、由 timeout 攔下,而非根本沒觸發
    assert any(d["state"] == "FALLING" for d in debug)


def test_slow_lie_down_no_event(cfg):
    # 緩慢躺下(如刻意躺地板):速度與角速度都低 → 不觸發。
    # 這是刻意的設計取捨(README 限制章節):慢速跌倒是規則法的已知盲區。
    df = make_trajectory([("stand", 1.0), ("to:lie", 5.0), ("lie", 2.0)], fps=FPS)
    events, _ = run_engine(df, FPS, cfg)
    assert events == []


def test_dropout_mid_lying_single_event(cfg):
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 4.0)], fps=FPS)
    df = drop_keypoints(df, 4.0, 4.3)  # 0.3s < max_kpt_gap_s:hold-last 撐住
    events, debug = run_engine(df, FPS, cfg, collect_debug=True)
    assert len(events) == 1
    # hold-last 必須真的在缺失期間維持狀態(而不是靠別的機制碰巧不裂):
    # 缺失窗內的 tick 應存在且全部維持 FALLEN/ALARM
    gap_states = [d["state"] for d in debug if 4.0 <= d["t_s"] < 4.3]
    assert gap_states and all(s in ("FALLEN", "ALARM") for s in gap_states)


def test_dropout_exceeds_ttl_truncates_without_ghost(cfg):
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 6.0)], fps=FPS)
    df = drop_keypoints(df, 4.5, 6.5)  # 2s 缺失 > TTL 與 track_lost_timeout
    events, _ = run_engine(df, FPS, cfg)
    assert len(events) == 1  # 事件被截斷,但缺失後的躺地不會生出幽靈第二事件
    # 下界鎖住 hold-last 的存在:有 hold-last 事件延續到 ~4.5+TTL(≈4.97),
    # 沒有 hold-last 會在 4.47 就斷——這 0.5s 是 TTL 機制的可觀測訊號
    assert events[0].end_time_s >= 4.8
    assert events[0].end_time_s <= 5.3  # 截在 hold-last 耗盡附近


def test_sentinel_conf_rows_no_crash(cfg):
    # keypoints.conf is None → 整列 -1.0:視同 dropout,不拋例外
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS)
    df = drop_keypoints(df, 2.2, 2.3, sentinel=-1.0)
    events, _ = run_engine(df, FPS, cfg)
    assert isinstance(events, list)


def test_unassigned_track_rows_ignored(cfg):
    # boxes.id is None → track_id = -1:不餵狀態機、不出事件、不崩潰
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS)
    df["track_id"] = np.int32(-1)
    events, debug = run_engine(df, FPS, cfg, collect_debug=True)
    assert events == []
    assert debug == []


def test_track_switch_stitched_into_one_event(cfg):
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS)
    df = switch_track(df, 2.35, new_id=7)  # 跌落中途斷 id(ByteTrack 常見)
    events, _ = run_engine(df, FPS, cfg)
    assert len(events) == 1
    assert events[0].track_ids == [1, 7]


def test_no_stitch_when_bboxes_far_apart(cfg):
    """負向縫合:視窗內出現的新 track 若 IoU 不足,必須各自獨立(不繼承狀態)。"""
    faller = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS, track_id=1)
    # 第二人的 track 於 t=2.5(跌落中途)才出生,且出生在遠處:
    # IoU=0,不得繼承任何人的 FSM
    late_walker = make_trajectory([("walk", 3.1)], fps=FPS, track_id=3, x0=900.0, seed=2)
    late_walker["frame_idx"] = late_walker["frame_idx"] + 75  # 從 t=2.5s 開始
    late_walker["t_ms"] = late_walker["t_ms"] + 2500.0
    df = (
        pd.concat([faller, late_walker], ignore_index=True)
        .sort_values(["frame_idx", "track_id"], kind="stable")
        .reset_index(drop=True)
    )
    events, _ = run_engine(df, FPS, cfg)
    assert len(events) == 1
    assert events[0].track_ids == [1]  # 走路的人沒有繼承跌倒者的狀態


def test_no_stitch_after_window_expires(cfg):
    """負向縫合:舊 track 消失超過 track_stitch_window_s 後,同位置的新 track 不縫合。"""
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 2.0)], fps=FPS, track_id=1)
    # 舊 track 於 t=4.6 結束;新 track 9 於 t=6.1 出現(間隔 1.5s > 1.0s 窗)
    reappear = make_trajectory([("lie", 2.0)], fps=FPS, track_id=9)
    reappear["frame_idx"] = reappear["frame_idx"] + 183  # t=6.1s 起
    reappear["t_ms"] = reappear["t_ms"] + 6100.0
    df = pd.concat([df, reappear], ignore_index=True).reset_index(drop=True)
    events, _ = run_engine(df, FPS, cfg)
    # 只有舊 track 的 ALARM 事件;新 track 是全新 FSM,靜躺不觸發
    assert len(events) == 1
    assert events[0].track_ids == [1]


def test_scale_invariance_exact(cfg):
    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS, noise=0.5)
    ev_a, _ = run_engine(df, FPS, cfg)
    ev_b, _ = run_engine(scale_coords(df, 0.5), FPS, cfg)  # ×0.5:浮點精確縮放
    assert [(e.start_frame, e.end_frame) for e in ev_a] == [
        (e.start_frame, e.end_frame) for e in ev_b
    ]


def test_fps_invariance(cfg):
    segs = [("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)]
    ev30, _ = run_engine(make_trajectory(segs, fps=30.0, noise=0.0), 30.0, cfg)
    ev15, _ = run_engine(make_trajectory(segs, fps=15.0, noise=0.0), 15.0, cfg)
    assert len(ev30) == len(ev15) == 1
    assert abs(ev30[0].start_time_s - ev15[0].start_time_s) < 0.2
    assert abs(ev30[0].end_time_s - ev15[0].end_time_s) < 0.2


def test_two_people_independent_states(cfg):
    """多人:一人跌倒、一人走動——事件只屬於跌倒者的 track。"""
    faller = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS, track_id=1)
    walker = make_trajectory([("walk", 5.6)], fps=FPS, track_id=2, x0=800.0, seed=1)
    df = (
        pd.concat([faller, walker], ignore_index=True)
        .sort_values(["frame_idx", "track_id"], kind="stable")
        .reset_index(drop=True)
    )
    events, _ = run_engine(df, FPS, cfg)
    assert len(events) == 1
    assert events[0].track_ids == [1]


def test_empty_input(cfg):
    df = pd.DataFrame(columns=CACHE_COLUMNS)
    events, debug = run_engine(df, FPS, cfg)
    assert events == [] and debug == []
