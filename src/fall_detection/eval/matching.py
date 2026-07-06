"""Event-level 評估:預測事件與 GT 事件的配對與指標計算。

協定(README 同步聲明):
- 預測事件與「GT 事件 ± tol」有任何時間交集即為候選配對;
- 每個 GT 以最大交集貪婪配一個預測(一對一);
- TP = 配對成功的 GT 數;FN = 未配對 GT;
  FP = fall 影片中未配對的預測 + ADL 影片中的所有預測
  (同一 GT 的重複預測算 FP:懲罰事件碎裂);
- 另報 video-level specificity(無任何預測事件的 ADL 影片比例),對齊文獻報法。
"""

from __future__ import annotations

from dataclasses import dataclass, field

Interval = tuple[float, float]


def interval_overlap(a: Interval, b: Interval) -> float:
    """兩區間的交集長度(秒);不相交為 0。"""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matches: list[tuple[Interval, Interval]] = field(default_factory=list)  # (pred, gt)


def match_events(
    preds: list[Interval], gts: list[Interval], tol_s: float = 0.5
) -> MatchResult:
    """單支影片內的貪婪一對一配對。"""
    used = [False] * len(preds)
    matches: list[tuple[Interval, Interval]] = []
    for gt in sorted(gts):
        expanded = (gt[0] - tol_s, gt[1] + tol_s)
        best_i, best_ov = None, 0.0
        for i, p in enumerate(preds):
            if used[i]:
                continue
            ov = interval_overlap(p, expanded)
            if ov > best_ov:
                best_i, best_ov = i, ov
        if best_i is not None and best_ov > 0.0:
            used[best_i] = True
            matches.append((preds[best_i], gt))
    tp = len(matches)
    fn = len(gts) - tp
    fp = sum(1 for u in used if not u)
    return MatchResult(tp=tp, fp=fp, fn=fn, matches=matches)


def evaluate_videos(videos: list[dict], tol_s: float = 0.5) -> dict:
    """彙整多支影片的 event-level 指標。

    Args:
        videos: 每支影片一個 dict:
            ``{"name": str, "is_adl": bool, "preds": [(s,e)...], "gts": [(s,e)...]}``
        tol_s: GT 區間兩端的容忍(秒)。

    Returns:
        指標 dict(precision/recall/f1 在分母為 0 時為 None,不假造數字),
        含 per-video 明細供失敗分析挑案例。
    """
    tp = fp = fn = 0
    adl_total = adl_clean = 0
    per_video = []
    for v in videos:
        res = match_events(list(v.get("preds", [])), list(v.get("gts", [])), tol_s)
        tp += res.tp
        fp += res.fp
        fn += res.fn
        if v.get("is_adl", False):
            adl_total += 1
            if not v.get("preds"):
                adl_clean += 1
        per_video.append(
            {
                "name": v["name"],
                "is_adl": bool(v.get("is_adl", False)),
                "tp": res.tp,
                "fp": res.fp,
                "fn": res.fn,
                "preds": [list(p) for p in v.get("preds", [])],
                "gts": [list(g) for g in v.get("gts", [])],
            }
        )

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    specificity = adl_clean / adl_total if adl_total > 0 else None

    return {
        "tol_s": tol_s,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "video_level_specificity": specificity,
        "n_videos": len(videos),
        "n_adl_videos": adl_total,
        "per_video": per_video,
    }
