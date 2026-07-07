"""影片 → keypoint cache(整條 pipeline 唯一需要 GPU 的步驟)。

批次模式 skip-existing:Colab 斷線重跑不重工。
每支影片各自 reset tracker,track id 不跨影片汙染。
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..io.cache import CACHE_COLUMNS, SCHEMA_VERSION, CacheMeta, write_cache
from ..io.video import iter_frames, probe
from .pose_tracker import PoseTracker


def _quick_sha1(path: str | Path, n_bytes: int = 1 << 20) -> str:
    """檔案前 1MB 的 sha1:足以偵測「換了影片但沒換 cache」的漂移,又不用讀全檔。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
        )
    except Exception:  # noqa: BLE001 - 沒 git(如 pip 安裝)不影響功能
        return ""


def extract_video(
    video_path: str | Path,
    out_path: str | Path,
    cfg: Config,
    device: str | None = None,
    tracker: PoseTracker | None = None,
    progress_every: int = 300,
) -> CacheMeta:
    """單支影片 → cache parquet。可傳入共用 tracker(批次時避免重複載模型)。"""
    video_path = Path(video_path)
    info = probe(video_path)

    if tracker is None:
        tracker = PoseTracker(
            model_name=cfg.model.name,
            tracker_yaml=cfg.model.tracker,
            conf=cfg.model.conf,
            iou=cfg.model.iou,
            device=device,
        )
    tracker.reset()

    rows = []
    n_frames = 0
    for frame_idx, frame in iter_frames(video_path):
        n_frames = frame_idx + 1
        det = tracker.track_frame(frame, frame_idx)
        t_ms = frame_idx / info.fps * 1000.0
        for i in range(det.n):
            rows.append(
                {
                    "frame_idx": np.int32(frame_idx),
                    "t_ms": t_ms,
                    "track_id": np.int32(det.track_ids[i]),
                    "bbox_x1": det.boxes[i, 0],
                    "bbox_y1": det.boxes[i, 1],
                    "bbox_x2": det.boxes[i, 2],
                    "bbox_y2": det.boxes[i, 3],
                    "bbox_conf": det.box_conf[i],
                    "kpts_xy": det.kpts_xy[i].reshape(-1),
                    "kpts_conf": det.kpts_conf[i],
                }
            )
        if progress_every and frame_idx % progress_every == 0 and frame_idx > 0:
            print(f"  {video_path.name}: {frame_idx} 幀…")

    df = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    meta = CacheMeta(
        schema_version=SCHEMA_VERSION,
        video_path=str(video_path),
        video_sha1=_quick_sha1(video_path),
        fps=info.fps,
        width=info.width,
        height=info.height,
        n_frames=n_frames,
        model_name=cfg.model.name,
        ultralytics_version=PoseTracker.ultralytics_version(),
        tracker_yaml=cfg.model.tracker,
        conf=cfg.model.conf,
        iou=cfg.model.iou,
        device=str(device),
        git_commit=_git_commit(),
    )
    write_cache(df, meta, out_path)
    return meta


def extract_batch(
    videos: list[str | Path],
    out_dir: str | Path,
    cfg: Config,
    device: str | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """批次抽取:輸出 {out_dir}/{影片檔名}.parquet;已存在即跳過(idempotent)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = PoseTracker(
        model_name=cfg.model.name,
        tracker_yaml=cfg.model.tracker,
        conf=cfg.model.conf,
        iou=cfg.model.iou,
        device=device,
    )
    outputs = []
    for i, video in enumerate(videos):
        video = Path(video)
        out_path = out_dir / f"{video.stem}.parquet"
        outputs.append(out_path)
        if skip_existing and out_path.exists():
            print(f"[{i + 1}/{len(videos)}] {video.stem}: skip(已存在)")
            continue
        print(f"[{i + 1}/{len(videos)}] {video.stem}: 抽取中…")
        extract_video(video, out_path, cfg, device=device, tracker=tracker)
    return outputs
