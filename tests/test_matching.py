"""Event-level 配對、指標與 URFD GT 解析的測試。"""

import io

import pandas as pd

from fall_detection.eval.ground_truth import gt_interval, load_gt_events
from fall_detection.eval.matching import evaluate_videos, interval_overlap, match_events


def test_overlap_basics():
    assert interval_overlap((0, 2), (1, 3)) == 1.0
    assert interval_overlap((0, 1), (2, 3)) == 0.0


def test_exact_and_partial_overlap_are_tp():
    res = match_events([(10.0, 12.0)], [(10.0, 12.0)], tol_s=0.5)
    assert (res.tp, res.fp, res.fn) == (1, 0, 0)
    res = match_events([(11.5, 14.0)], [(10.0, 12.0)], tol_s=0.5)
    assert (res.tp, res.fp, res.fn) == (1, 0, 0)


def test_tolerance_boundary():
    # GT [10,12]+tol0.5 → [9.5,12.5];預測 [12.3,12.6] 交集 0.2 > 0 → TP
    res = match_events([(12.3, 12.6)], [(10.0, 12.0)], tol_s=0.5)
    assert res.tp == 1
    # 預測 [12.6,13.0] 與 [9.5,12.5] 無交集 → FP + FN
    res = match_events([(12.6, 13.0)], [(10.0, 12.0)], tol_s=0.5)
    assert (res.tp, res.fp, res.fn) == (0, 1, 1)


def test_duplicate_predictions_penalized():
    # 一個 GT、兩個都重疊的預測:只配一個,另一個算 FP(懲罰事件碎裂)
    res = match_events([(10.0, 10.8), (11.2, 12.0)], [(10.0, 12.0)], tol_s=0.5)
    assert (res.tp, res.fp, res.fn) == (1, 1, 0)


def test_greedy_prefers_larger_overlap():
    res = match_events([(9.9, 10.2), (10.5, 12.0)], [(10.0, 12.0)], tol_s=0.0)
    assert res.matches[0][0] == (10.5, 12.0)


def test_one_prediction_cannot_match_two_gts():
    # 一對一約束:一個橫跨兩個 GT 的長預測只能配到一個,另一個 GT 記 FN
    res = match_events([(2.0, 9.0)], [(2.5, 3.5), (7.0, 8.0)], tol_s=0.5)
    assert (res.tp, res.fp, res.fn) == (1, 0, 1)


def test_two_gts_two_preds_both_match():
    res = match_events([(2.4, 3.6), (6.9, 8.2)], [(2.5, 3.5), (7.0, 8.0)], tol_s=0.5)
    assert (res.tp, res.fp, res.fn) == (2, 0, 0)


def test_evaluate_videos_aggregation_and_specificity():
    videos = [
        {"name": "fall-01", "is_adl": False, "preds": [(2.0, 5.0)], "gts": [(2.1, 4.0)]},
        {"name": "fall-02", "is_adl": False, "preds": [], "gts": [(3.0, 5.0)]},
        {"name": "adl-01", "is_adl": True, "preds": [(1.0, 2.0)], "gts": []},
        {"name": "adl-02", "is_adl": True, "preds": [], "gts": []},
    ]
    m = evaluate_videos(videos, tol_s=0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and abs(m["f1"] - 0.5) < 1e-9
    assert m["video_level_specificity"] == 0.5


def test_no_predictions_metrics_are_none_not_fake():
    m = evaluate_videos([{"name": "adl-01", "is_adl": True, "preds": [], "gts": []}])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None


def test_gt_interval_convention():
    # -1 ×5, 0 ×3, 1 ×4, -1 ×2 → 事件 = [第一個 0, 連續 1 區段末幀] = (6, 12)
    labels = [-1] * 5 + [0] * 3 + [1] * 4 + [-1] * 2
    df = pd.DataFrame({"sequence": "fall-01", "frame": range(1, 15), "label": labels})
    assert gt_interval(df) == (6, 12)


def test_gt_interval_adl_returns_none():
    df = pd.DataFrame({"sequence": "adl-01", "frame": range(1, 8), "label": [-1] * 7})
    assert gt_interval(df) is None


def test_load_gt_events_from_csv(tmp_path):
    rows = []
    for i in range(1, 11):
        label = -1 if i <= 4 else (0 if i <= 6 else 1)
        rows.append(f"fall-01,{i},{label},0,0,0,0,0,0,0,0")
    for i in range(1, 6):
        rows.append(f"adl-01,{i},-1,0,0,0,0,0,0,0,0")
    p = tmp_path / "urfall-cam0-falls.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    events = load_gt_events(p)
    assert events == {"fall-01": (5, 10)}
