"""tune split 網格調參 → 挑選最佳設定 → test split 指標。

只吃 pandas/numpy(透過 ``rules.run_engine``),不碰 torch/ultralytics/cv2:
調參本身複用 GPU/規則引擎解耦的設計,在 CPU runtime(甚至本機)就能跑。

網格刻意保持小(每個參數 3~4 個候選值):tune split 只有 23 支影片,
組合太多容易在小樣本上過擬合出「剛好適合這 23 支」而非真正較好的設定
——寧可保守也不要虛胖的網格製造虛假的信心。
"""

from __future__ import annotations

import itertools

import pandas as pd

from ..config import Config
from ..eval.matching import evaluate_videos, match_events
from ..rules import run_engine


def _events_to_pred_intervals(events) -> list[tuple[float, float]]:
    return [(e.start_time_s, e.end_time_s) for e in events]


# model.conf(偵測信心門檻)在 extract 階段就已經套用、烘進 cache 裡(過門檻
# 的偵測才會落成 cache row),事後重跑規則引擎無法模擬「當初用了不同 conf」
# ——要調 conf 得重新跑一次 GPU extract。kpt_conf_min 則相反:cache 存的是
# 逐關鍵點的原始 conf,門檻是規則引擎讀 cache 時才套用,可以放心事後調參。
_MODEL_FIELDS = {"kpt_conf_min"}


def _apply_params(base_cfg: Config, params: dict) -> Config:
    """把一組候選參數套進 base_cfg 的複本(model 與 rules 欄位分開處理)。"""
    cfg = base_cfg.model_copy(deep=True)
    rule_updates = {k: v for k, v in params.items() if k not in _MODEL_FIELDS}
    model_updates = {k: v for k, v in params.items() if k in _MODEL_FIELDS}
    if rule_updates:
        cfg.rules = cfg.rules.model_copy(update=rule_updates)
    if model_updates:
        cfg.model = cfg.model.model_copy(update=model_updates)
    return cfg


def make_param_grid(
    kpt_conf_min_values: list[float],
    v_fall_enter_values: list[float],
    theta_lying_enter_values: list[float],
    t_confirm_fallen_s_values: list[float],
    window_confirm_s_values: list[float],
    theta_hysteresis_gap: float = 20.0,
) -> list[dict]:
    """建構本專案 M4 使用的參數網格。

    ``theta_upright_exit`` 隨 ``theta_lying_enter`` 連動(固定遲滯間距),
    而不是獨立笛卡兒積出來的自由參數——否則容易生出違反遲滯一致性
    (exit >= enter)的組合,或製造原本不存在、只因為間距怪異而變好/變差的假象。

    只網格搜尋規則引擎階段仍可調的參數(見上方 ``_MODEL_FIELDS`` 註解);
    ``model.conf`` 已烘進 cache,不在這個網格裡。

    ``window_confirm_s``(躺姿投票時窗)是第二輪才加入的維度:第一輪調參
    發現多支跌倒影片在 FALLING 觸發、姿態也已達標之後,track 就在時窗投票
    還沒累積夠樣本前消失(見 state_machine.finalize 對這個情境的收尾規則)。
    縮小這個時窗能讓確認更快發生,直接針對這個失敗模式;但窗太小也會讓
    單幀雜訊更容易誤觸發,因此仍是候選網格的一員而非直接調小,讓資料自己說話。
    """
    combos = []
    for kpt_conf_min, v, theta, tconf, wconf in itertools.product(
        kpt_conf_min_values,
        v_fall_enter_values,
        theta_lying_enter_values,
        t_confirm_fallen_s_values,
        window_confirm_s_values,
    ):
        combos.append(
            {
                "kpt_conf_min": kpt_conf_min,
                "v_fall_enter": v,
                "theta_lying_enter": theta,
                "theta_upright_exit": theta - theta_hysteresis_gap,
                "t_confirm_fallen_s": tconf,
                "window_confirm_s": wconf,
            }
        )
    return combos


def build_video_dicts(
    cache_by_seq: dict[str, tuple[pd.DataFrame, object]],
    sequences: list[str],
    adl_seqs: set[str],
    gt_by_seq: dict[str, tuple[int, int]],
    cfg: Config,
) -> list[dict]:
    """對每支序列跑一次 run_engine,組成 ``evaluate_videos`` 需要的 per-video dict。

    ADL 影片一律 gts=[](即使該影片標註本身有躺姿片段,那是刻意的日常臥床
    等動作、不是跌倒;任何預測事件都算 FP,見 eval.matching 的協定)。
    """
    videos = []
    for seq in sequences:
        df, meta = cache_by_seq[seq]
        events, _ = run_engine(df, meta.fps, cfg)
        is_adl = seq in adl_seqs
        gts: list[tuple[float, float]] = []
        if not is_adl and seq in gt_by_seq:
            start_f, end_f = gt_by_seq[seq]
            gts = [(start_f / meta.fps, end_f / meta.fps)]
        videos.append(
            {
                "name": seq,
                "is_adl": is_adl,
                "preds": _events_to_pred_intervals(events),
                "gts": gts,
            }
        )
    return videos


def grid_search(
    cache_by_seq: dict[str, tuple[pd.DataFrame, object]],
    sequences: list[str],
    adl_seqs: set[str],
    gt_by_seq: dict[str, tuple[int, int]],
    base_cfg: Config,
    param_combos: list[dict],
    tol_s: float = 0.5,
) -> list[dict]:
    """對 ``param_combos`` 中每一組跑一次 ``evaluate_videos``。

    回傳 ``[{"params": dict, "metrics": dict}, ...]``,``metrics`` 含 per_video
    明細,選出最佳組合後仍可回頭做失敗分析,不必重跑。
    """
    results = []
    for params in param_combos:
        cfg = _apply_params(base_cfg, params)
        videos = build_video_dicts(cache_by_seq, sequences, adl_seqs, gt_by_seq, cfg)
        metrics = evaluate_videos(videos, tol_s=tol_s)
        results.append({"params": params, "metrics": metrics})
    return results


def select_best(results: list[dict], min_precision: float = 0.5) -> dict:
    """挑選規則:recall 優先,precision 需 >= min_precision 才列入候選。

    理由:漏掉真實跌倒的代價(沒人去查看、錯過黃金救援時間)高於多一次誤報,
    因此以 recall 為主要目標;但完全不設下限會讓「每一幀都報跌倒」這種
    退化解也拿到 recall=1.0,所以用 min_precision 擋掉明顯退化的組合。

    若沒有任何組合達到 min_precision,誠實地退回選 precision 最高者,
    而不是假裝有組合達標。
    """

    def _recall(r: dict) -> float:
        m = r["metrics"]
        return m["recall"] if m["recall"] is not None else -1.0

    def _precision(r: dict) -> float:
        m = r["metrics"]
        return m["precision"] if m["precision"] is not None else -1.0

    eligible = [r for r in results if _precision(r) >= min_precision]
    if eligible:
        return max(eligible, key=lambda r: (_recall(r), _precision(r)))
    # 沒人達標:已經在「大家 precision 都不夠」的處境,此時繼續以 recall 為
    # 優先排序沒有意義(只會挑到最退化的組合);改選最接近達標的 precision。
    return max(results, key=lambda r: (_precision(r), _recall(r)))


def list_failure_cases(metrics: dict) -> dict:
    """從 ``evaluate_videos`` 的輸出中挑出有 FP/FN 的影片,並算出實際沒配對到
    的預測/GT 區間(供人工複核、挑選失敗分析案例)。"""
    fp_cases = []
    fn_cases = []
    for v in metrics["per_video"]:
        if v["fp"] == 0 and v["fn"] == 0:
            continue
        preds = [tuple(p) for p in v["preds"]]
        gts = [tuple(g) for g in v["gts"]]
        res = match_events(preds, gts, tol_s=metrics["tol_s"])
        matched_preds = {p for p, _ in res.matches}
        matched_gts = {g for _, g in res.matches}
        unmatched_preds = [p for p in preds if p not in matched_preds]
        unmatched_gts = [g for g in gts if g not in matched_gts]
        if unmatched_preds:
            fp_cases.append({"name": v["name"], "is_adl": v["is_adl"], "fp_intervals": unmatched_preds})
        if unmatched_gts:
            fn_cases.append({"name": v["name"], "fn_intervals": unmatched_gts})
    return {"fp_cases": fp_cases, "fn_cases": fn_cases}
