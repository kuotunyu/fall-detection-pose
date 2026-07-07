"""M4 網格調參/選擇邏輯的測試(合成軌跡,不碰真實 cache/影片)。"""

from types import SimpleNamespace

from fall_detection.eval.report import (
    _apply_params,
    build_video_dicts,
    grid_search,
    list_failure_cases,
    make_param_grid,
    select_best,
)

from synthetic import make_trajectory

FPS = 30.0


def _cache_entry(df):
    return df, SimpleNamespace(fps=FPS)


def test_make_param_grid_keeps_hysteresis_valid():
    grid = make_param_grid([0.35], [1.0, 1.5], [50, 60], [1.0], [0.4])
    assert len(grid) == 1 * 2 * 2 * 1 * 1
    for combo in grid:
        assert combo["theta_upright_exit"] < combo["theta_lying_enter"]
        assert "conf" not in combo  # model.conf 已烘進 cache,不應出現在這個網格裡


def test_apply_params_routes_kpt_conf_min_to_model_not_rules(cfg):
    """迴歸測試:kpt_conf_min 屬於 ModelConfig,若誤塞進 RulesConfig.model_copy,
    pydantic 不會報錯,只會靜靜掛成一個沒人讀的幽靈屬性(model_dump 也看不到)
    ——網格搜尋會「看起來」跑過這個維度,實際上完全沒生效。"""
    new_cfg = _apply_params(cfg, {"kpt_conf_min": 0.9, "v_fall_enter": 2.0})
    assert new_cfg.model.kpt_conf_min == 0.9
    assert new_cfg.rules.v_fall_enter == 2.0
    assert "kpt_conf_min" not in new_cfg.rules.model_dump()


def test_build_video_dicts_fall_and_adl(cfg):
    fall_df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS)
    walk_df = make_trajectory([("walk", 5.0)], fps=FPS, seed=1)
    cache_by_seq = {"fall-01": _cache_entry(fall_df), "adl-01": _cache_entry(walk_df)}
    gt_by_seq = {"fall-01": (60, 90)}  # 幀號隨意,只測試轉換邏輯

    videos = build_video_dicts(
        cache_by_seq, ["fall-01", "adl-01"], adl_seqs={"adl-01"}, gt_by_seq=gt_by_seq, cfg=cfg
    )
    by_name = {v["name"]: v for v in videos}
    assert by_name["fall-01"]["is_adl"] is False
    assert by_name["fall-01"]["gts"] == [(2.0, 3.0)]  # 60/30, 90/30
    assert len(by_name["fall-01"]["preds"]) == 1  # 教科書跌倒恰一事件

    assert by_name["adl-01"]["is_adl"] is True
    assert by_name["adl-01"]["gts"] == []  # ADL 一律不當跌倒 GT
    assert by_name["adl-01"]["preds"] == []  # 走路不觸發


def test_grid_search_stricter_thresholds_suppress_detection(cfg):
    """v_fall_enter/omega_enter 都調到不可能觸發的高值 → recall 應降到 0,
    驗證 grid_search 確實用了每組候選參數(而不是重用同一個 cfg)。"""
    fall_df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=FPS)
    cache_by_seq = {"fall-01": _cache_entry(fall_df)}
    gt_by_seq = {"fall-01": (60, 165)}

    combos = [
        {"kpt_conf_min": 0.35, "v_fall_enter": 1.5, "theta_lying_enter": 60, "theta_upright_exit": 40, "t_confirm_fallen_s": 1.0},
        {"kpt_conf_min": 0.35, "v_fall_enter": 10.0, "omega_enter": 10000.0, "theta_lying_enter": 60, "theta_upright_exit": 40, "t_confirm_fallen_s": 1.0},
    ]
    results = grid_search(
        cache_by_seq, ["fall-01"], adl_seqs=set(), gt_by_seq=gt_by_seq, base_cfg=cfg, param_combos=combos
    )
    assert len(results) == 2
    recalls = {r["params"]["v_fall_enter"]: r["metrics"]["recall"] for r in results}
    assert recalls[1.5] == 1.0  # 正常閾值抓得到
    assert recalls[10.0] == 0.0  # 兩個觸發路徑都關到不可能觸發,理所當然抓不到


def test_select_best_prefers_recall_within_precision_floor():
    results = [
        {"params": {"a": 1}, "metrics": {"recall": 0.9, "precision": 0.3}},
        {"params": {"a": 2}, "metrics": {"recall": 0.7, "precision": 0.6}},
        {"params": {"a": 3}, "metrics": {"recall": 0.5, "precision": 0.9}},
    ]
    best = select_best(results, min_precision=0.5)
    assert best["params"]["a"] == 2  # 達標(precision>=0.5)中 recall 較高的那組


def test_select_best_falls_back_when_none_meet_floor():
    results = [
        {"params": {"a": 1}, "metrics": {"recall": 0.9, "precision": 0.1}},
        {"params": {"a": 2}, "metrics": {"recall": 0.5, "precision": 0.4}},
    ]
    best = select_best(results, min_precision=0.5)
    assert best["params"]["a"] == 2  # 沒人達標:誠實退回 precision 最高者


def test_list_failure_cases_identifies_unmatched_intervals():
    metrics = {
        "tol_s": 0.5,
        "per_video": [
            {"name": "fall-01", "is_adl": False, "tp": 1, "fp": 0, "fn": 0,
             "preds": [[2.0, 3.0]], "gts": [[2.1, 3.1]]},
            {"name": "fall-02", "is_adl": False, "tp": 0, "fp": 0, "fn": 1,
             "preds": [], "gts": [[5.0, 6.0]]},
            {"name": "adl-05", "is_adl": True, "tp": 0, "fp": 1, "fn": 0,
             "preds": [[1.0, 2.0]], "gts": []},
        ],
    }
    out = list_failure_cases(metrics)
    assert [c["name"] for c in out["fn_cases"]] == ["fall-02"]
    assert out["fn_cases"][0]["fn_intervals"] == [(5.0, 6.0)]
    assert [c["name"] for c in out["fp_cases"]] == ["adl-05"]
    assert out["fp_cases"][0]["fp_intervals"] == [(1.0, 2.0)]
