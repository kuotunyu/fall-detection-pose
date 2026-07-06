"""規則引擎:keypoint cache rows → 每 track 特徵管線 → 狀態機 → 跌倒事件。

職責:
1. 逐幀分流:cache 的每列依 track_id 分派到該 track 的特徵管線 + 狀態機;
   ``track_id = -1``(未確認偵測)不餵狀態機。
2. 特徵管線:平滑(滑動中位數)、以軀幹長 L̃ 正規化、固定「時間窗」差分
   算垂直速度與角速度(尺度與 fps 不變性都在這一層達成)。
3. keypoint dropout 容忍:無效幀 hold-last(TTL = ``max_kpt_gap_s``),
   逾時凍結;連續無有效觀測超過 ``track_lost_timeout_s`` 即終結該 track。
4. track 縫合:ByteTrack 在跌倒瞬間常斷 id——新 track 出現時,若與剛消失的
   舊 track 末 bbox 的 IoU 夠高,直接繼承舊狀態機(不魔改 tracker)。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import Config
from ..events.schema import FallEvent, postprocess_events
from .features import compute_frame_geometry
from .smoothing import RollingMedian, TimedBuffer
from .state_machine import FallStateMachine, TickInput

# L̃(軀幹長滑動中位數)的時窗:遠長於單次跌倒(~1s),
# 朝鏡頭跌倒造成的軀幹投影縮短不會即刻拉低尺度基準。
TORSO_MEDIAN_WINDOW_S = 2.0


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _TrackRunner:
    """單一 track 的特徵管線(平滑/正規化/差分)+ 其狀態機。

    縫合時只繼承狀態機:平滑緩衝重新起算(幾幀內即填滿,
    代價遠小於把兩條 track 的緩衝硬接在一起的複雜度)。
    """

    def __init__(self, cfg: Config, fps: float, fsm: FallStateMachine):
        r = cfg.rules
        smooth_n = max(1, round(fps * r.smooth_s))
        self.theta_med = RollingMedian(smooth_n)
        self.aspect_med = RollingMedian(smooth_n)
        self.hhip_med = RollingMedian(smooth_n)
        self.hipy_med3 = RollingMedian(3)
        self.torso_med = RollingMedian(max(1, round(fps * TORSO_MEDIAN_WINDOW_S)))
        horizon = r.velocity_window_s * 1.5 + 0.2
        self.hipy_buf = TimedBuffer(horizon)
        self.theta_buf = TimedBuffer(horizon)
        self.fsm = fsm
        self.last_t: float | None = None
        self.last_valid_t: float | None = None
        self.last_bbox: tuple | None = None
        self.held: TickInput | None = None

    def step(
        self,
        cfg: Config,
        track_id: int,
        frame_idx: int,
        t_s: float,
        bbox: tuple,
        kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
        debug: list | None,
    ) -> None:
        r = cfg.rules
        geo = compute_frame_geometry(kpts_xy, kpts_conf, np.asarray(bbox), cfg.model.kpt_conf_min)

        if geo.valid:
            l_tilde = self.torso_med.push(geo.torso_len)
            theta_s = self.theta_med.push(geo.theta_deg)
            aspect_s = (
                self.aspect_med.push(geo.bbox_aspect)
                if not math.isnan(geo.bbox_aspect)
                else float("nan")
            )
            h_hip = None
            if geo.ankle_valid and l_tilde > 0:
                h_hip = self.hhip_med.push(geo.hip_ankle_gap / l_tilde)
            hip_y_f = self.hipy_med3.push(geo.hip_y)
            # 差分先取歷史再 push 當前值(否則 Δ 內含自身)
            v_raw = self.hipy_buf.rate(t_s, hip_y_f, r.velocity_window_s)
            self.hipy_buf.push(t_s, hip_y_f)
            omega = self.theta_buf.rate(t_s, theta_s, r.velocity_window_s)
            self.theta_buf.push(t_s, theta_s)
            v_norm = (v_raw / l_tilde) if (v_raw is not None and l_tilde > 0) else None

            tick = TickInput(
                t_s=t_s,
                frame_idx=frame_idx,
                theta_deg=theta_s,
                bbox_aspect=aspect_s,
                h_hip=h_hip,
                v_norm=v_norm,
                omega=omega,
            )
            self.held = tick
            self.last_valid_t = t_s
            self.fsm.tick(tick)
        elif (
            self.held is not None
            and self.last_valid_t is not None
            and (t_s - self.last_valid_t) <= r.max_kpt_gap_s
        ):
            # hold-last:姿態沿用上次有效值,速度/角速度歸零(缺測不得觸發新事件,
            # 但已在 FALLEN/ALARM 的狀態得以維持,dropout 不會把一次跌倒切成兩段)
            tick = TickInput(
                t_s=t_s,
                frame_idx=frame_idx,
                theta_deg=self.held.theta_deg,
                bbox_aspect=self.held.bbox_aspect,
                h_hip=self.held.h_hip,
                v_norm=0.0,
                omega=0.0,
            )
            self.fsm.tick(tick)
        # else:逾 TTL → 凍結(不 tick);終結與否由引擎依 track_lost_timeout_s 決定

        self.last_t = t_s
        self.last_bbox = tuple(float(v) for v in bbox)
        if debug is not None:
            debug.append(
                {
                    "frame_idx": int(frame_idx),
                    "track_id": int(track_id),
                    "t_s": round(t_s, 4),
                    "valid": bool(geo.valid),
                    "theta_deg": round(self.held.theta_deg, 2) if self.held else None,
                    "bbox_aspect": (
                        round(self.held.bbox_aspect, 3)
                        if self.held and not math.isnan(self.held.bbox_aspect)
                        else None
                    ),
                    "h_hip": (
                        round(self.held.h_hip, 3)
                        if self.held and self.held.h_hip is not None
                        else None
                    ),
                    "v_norm": (
                        round(self.held.v_norm, 3)
                        if geo.valid and self.held and self.held.v_norm is not None
                        else None
                    ),
                    "state": self.fsm.state.value,
                }
            )


def _pop_stitch_source(
    runners: dict[int, _TrackRunner], bbox: tuple, t_s: float, cfg: Config
) -> _TrackRunner | None:
    """在「最近消失」的 runner 中找縫合對象;找到即自 runners 移除並回傳。

    只考慮本幀沒被更新的 runner(仍活躍的 track 不是縫合對象)。
    """
    r = cfg.rules
    best_tid, best_iou = None, 0.0
    for tid, ru in runners.items():
        if ru.last_t is None or ru.last_bbox is None:
            continue
        if ru.last_t >= t_s - 1e-9:  # 本幀已更新:仍活著
            continue
        if t_s - ru.last_t > r.track_stitch_window_s:
            continue
        iou = _iou(ru.last_bbox, bbox)
        if iou >= r.track_stitch_iou and iou > best_iou:
            best_tid, best_iou = tid, iou
    if best_tid is None:
        return None
    return runners.pop(best_tid)


def run_engine(
    df: pd.DataFrame, fps: float, cfg: Config, collect_debug: bool = False
) -> tuple[list[FallEvent], list[dict]]:
    """對一支影片的 cache rows 執行規則引擎。

    Args:
        df: keypoint cache rows(欄位見 ``io.cache.CACHE_COLUMNS``)。
        fps: 影片幀率(取自 cache meta;所有時間計算的基準)。
        cfg: 完整設定。
        collect_debug: 是否回傳 per-frame per-track 特徵紀錄(失敗分析用)。

    Returns:
        (後處理完成的事件列表, debug 紀錄列表)
    """
    debug: list[dict] | None = [] if collect_debug else None
    if df.empty or fps <= 0:
        return [], (debug or [])

    r = cfg.rules
    runners: dict[int, _TrackRunner] = {}
    events_raw: list[FallEvent] = []

    df = df.sort_values(["frame_idx", "track_id"], kind="stable")
    for _, grp in df.groupby("frame_idx", sort=True):
        rows = list(grp.itertuples(index=False))
        t_frame = float(rows[0].t_ms) / 1000.0

        # 先更新既有 track,再處理新 track:避免把「本幀仍活著的 track」誤當縫合對象
        existing = [row for row in rows if int(row.track_id) in runners]
        newcomers = [
            row for row in rows if int(row.track_id) >= 0 and int(row.track_id) not in runners
        ]

        for row in existing:
            tid = int(row.track_id)
            t_s = float(row.t_ms) / 1000.0
            runner = runners[tid]
            if (
                runner.last_valid_t is not None
                and (t_s - runner.last_valid_t) > r.track_lost_timeout_s
            ):
                # 同一 id 長時間無有效觀測後重現:舊片段終結,重新起算
                events_raw.extend(runner.fsm.finalize())
                runner = _TrackRunner(cfg, fps, FallStateMachine(r, tid))
                runners[tid] = runner
            runner.step(cfg, tid, int(row.frame_idx), t_s, _bbox(row), row.kpts_xy, row.kpts_conf, debug)

        for row in newcomers:
            tid = int(row.track_id)
            t_s = float(row.t_ms) / 1000.0
            bbox = _bbox(row)
            src = _pop_stitch_source(runners, bbox, t_s, cfg)
            if src is not None:
                src.fsm.adopt(tid)
                runner = _TrackRunner(cfg, fps, src.fsm)
                # 縫合時繼承尺度基準與 hold-last 狀態:
                # L̃ 若在跌倒瞬間重新定錨到「投影縮短的軀幹」,正規化就失真
                runner.torso_med = src.torso_med
                runner.held = src.held
                runner.last_valid_t = src.last_valid_t
            else:
                runner = _TrackRunner(cfg, fps, FallStateMachine(r, tid))
            runners[tid] = runner
            runner.step(cfg, tid, int(row.frame_idx), t_s, bbox, row.kpts_xy, row.kpts_conf, debug)

        # 清掃:超過縫合窗且超過 lost timeout 的 runner 一律終結
        stale_cut = t_frame - max(r.track_lost_timeout_s, r.track_stitch_window_s)
        for tid in [t for t, ru in runners.items() if ru.last_t is not None and ru.last_t < stale_cut]:
            events_raw.extend(runners.pop(tid).fsm.finalize())

    for runner in runners.values():
        events_raw.extend(runner.fsm.finalize())

    return postprocess_events(events_raw, cfg.events), (debug or [])


def _bbox(row) -> tuple:
    return (
        float(row.bbox_x1),
        float(row.bbox_y1),
        float(row.bbox_x2),
        float(row.bbox_y2),
    )
