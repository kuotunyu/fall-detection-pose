"""影片讀寫封裝(依賴 opencv/imageio-ffmpeg;extras: infer)。

輸出約定:所有「給人看」的影片一律 H.264(yuv420p + faststart)。
OpenCV 的 mp4v(MPEG-4 Part 2)在瀏覽器/Gradio 會是黑畫面,且 pip 版
opencv-python 因授權不含 H.264 encoder——因此先寫 mp4v 暫存檔,
關檔時用 ffmpeg(由 imageio-ffmpeg 保證存在)重編碼。
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    n_frames: int


def probe(path: str | Path) -> VideoInfo:
    """取得影片基本資訊;開不了檔直接拋錯。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片:{path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """逐幀迭代 (frame_idx, BGR frame)。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片:{path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def get_ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def reencode_h264(src: str | Path, dst: str | Path) -> None:
    """重編碼為瀏覽器可播的 H.264 mp4(yuv420p + faststart)。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",       # yuv444 在 Chrome/Safari 播不了
        "-movflags", "+faststart",   # moov atom 前置,邊下載邊播
        str(dst),
    ]
    subprocess.run(cmd, check=True)


class H264VideoWriter:
    """先寫 mp4v 暫存,close() 時 ffmpeg 重編碼成 H.264 輸出。"""

    def __init__(self, out_path: str | Path, fps: float, width: int, height: int):
        self.out_path = Path(out_path)
        self._tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._tmp), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"VideoWriter 開啟失敗(fps={fps}, size={width}x{height})")

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def close(self) -> None:
        self._writer.release()
        try:
            reencode_h264(self._tmp, self.out_path)
        finally:
            self._tmp.unlink(missing_ok=True)

    def __enter__(self) -> "H264VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_video_mp4v(frames: Iterator[np.ndarray], out_path: str | Path, fps: float) -> int:
    """把幀序列寫成 mp4v 影片(僅供內部再處理,如 URFD PNG 序列重組;
    非瀏覽器播放用途,不做 H.264 重編碼以省時)。回傳寫入幀數。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    for frame in frames:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter 開啟失敗:{out_path}")
        writer.write(frame)
        n += 1
    if writer is not None:
        writer.release()
    return n
