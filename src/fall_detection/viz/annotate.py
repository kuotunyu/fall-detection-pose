"""標註影片輸出:骨架 + track id + 狀態標籤 + ALARM 警示。

狀態來源是規則引擎的 per-frame debug 紀錄(單一事實來源:
畫面上看到的狀態就是引擎當下判定的狀態,而非事後由事件區間反推),
ALARM 橫幅出現的時刻即真實告警延遲,demo 不美化。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..events.schema import FallEvent
from ..io.cache import N_KPTS
from ..io.video import H264VideoWriter, iter_frames, probe

# COCO 17 骨架連線(肢段)
SKELETON = [
    (5, 7), (7, 9),      # 左臂
    (6, 8), (8, 10),     # 右臂
    (5, 6),              # 肩線
    (5, 11), (6, 12),    # 軀幹
    (11, 12),            # 髖線
    (11, 13), (13, 15),  # 左腿
    (12, 14), (14, 16),  # 右腿
    (0, 5), (0, 6),      # 頭-肩
]

STATE_COLORS = {  # BGR
    "UPRIGHT": (80, 200, 80),
    "FALLING": (0, 165, 255),
    "FALLEN": (0, 60, 255),
    "ALARM": (0, 0, 255),
}


def _draw_person(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    bbox: tuple,
    track_id: int,
    state: str,
    kpt_conf_min: float,
) -> None:
    color = STATE_COLORS.get(state, (200, 200, 200))
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    pts = kpts_xy.reshape(N_KPTS, 2)
    visible = kpts_conf >= kpt_conf_min
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            pa = tuple(int(v) for v in pts[a])
            pb = tuple(int(v) for v in pts[b])
            cv2.line(frame, pa, pb, color, 2)
    for i in range(N_KPTS):
        if visible[i]:
            cv2.circle(frame, tuple(int(v) for v in pts[i]), 3, color, -1)

    label = f"id {track_id}  {state}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 8, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )


def _draw_alarm_banner(frame: np.ndarray, track_ids: list[int]) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 46), (0, 0, 220), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    ids = ",".join(str(t) for t in sorted(set(track_ids)))
    cv2.putText(
        frame,
        f"FALL ALARM  (track {ids})",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )


def annotate_video(
    video_path: str | Path,
    cache_df: pd.DataFrame,
    fps: float,
    cfg: Config,
    events: list[FallEvent],
    debug_records: list[dict],
    out_path: str | Path,
) -> Path:
    """輸出標註影片(H.264)。

    Args:
        cache_df: keypoint cache rows(骨架繪製來源)。
        events: 引擎輸出的事件(用於畫面右上角事件計數)。
        debug_records: ``run_engine(collect_debug=True)`` 的 per-frame 狀態。
    """
    out_path = Path(out_path)
    info = probe(video_path)

    state_by_frame_track: dict[tuple[int, int], str] = {
        (d["frame_idx"], d["track_id"]): d["state"] for d in debug_records
    }
    rows_by_frame: dict[int, list] = {}
    for row in cache_df.itertuples(index=False):
        rows_by_frame.setdefault(int(row.frame_idx), []).append(row)

    with H264VideoWriter(out_path, info.fps, info.width, info.height) as writer:
        for frame_idx, frame in iter_frames(video_path):
            t_s = frame_idx / fps
            alarm_tracks: list[int] = []
            for row in rows_by_frame.get(frame_idx, []):
                tid = int(row.track_id)
                state = state_by_frame_track.get((frame_idx, tid), "UPRIGHT")
                if tid < 0:
                    state = "-"  # 未指派 track 的偵測:只畫框不標狀態
                _draw_person(
                    frame,
                    np.asarray(row.kpts_xy),
                    np.asarray(row.kpts_conf),
                    (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
                    tid,
                    state,
                    cfg.model.kpt_conf_min,
                )
                if state == "ALARM":
                    alarm_tracks.append(tid)
            if alarm_tracks:
                _draw_alarm_banner(frame, alarm_tracks)
            n_done = sum(1 for e in events if e.end_time_s <= t_s)
            cv2.putText(
                frame,
                f"frame {frame_idx}  t={t_s:6.2f}s  events={n_done}/{len(events)}",
                (10, info.height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            writer.write(frame)
    return out_path
