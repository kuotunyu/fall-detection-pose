"""模型 x 裝置 FPS benchmark(見 notebooks/04_benchmark.ipynb)。

方法論:
- 固定影片先全部解碼進記憶體,benchmark 計時不受磁碟/解碼 I/O 影響。
- 每輪 warmup(預設 20 幀,不計時)後才開始量測,排除模型/CUDA 初始化開銷。
- GPU 計時前後夾 ``torch.cuda.synchronize()``:CUDA 呼叫預設非同步,
  不同步計時會量到「排隊時間」而非真正的運算時間。
- 每個設定跑 ``n_runs``(預設 3)輪、取 FPS 中位數,抗單輪雜訊(背景任務、
  暖機不完全等)。
- 純推論與端到端只呼叫一次 ``model.track()``:同一幀被 ``persist=True``
  的 tracker 吃兩次會弄亂 track 狀態,因此端到端的「轉換」耗時是在同一次
  呼叫後另外計時,而非重跑一次 track()(見 ``inference.pose_tracker.convert_results``)。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..inference.pose_tracker import PoseTracker, convert_results

DEFAULT_N_FRAMES = 300
DEFAULT_WARMUP = 20
DEFAULT_N_RUNS = 3


@dataclass
class BenchResult:
    model_name: str
    device: str
    quantize: str | None
    n_frames: int
    n_runs: int
    pure_inference_fps: float
    end_to_end_fps: float
    p50_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_frames(video_path: str, n_frames: int = DEFAULT_N_FRAMES) -> list[np.ndarray]:
    """先把幀序列全部解碼進記憶體;若影片幀數 < n_frames,回傳全部可用幀
    (用多少算多少,不假裝湊滿——BenchResult.n_frames 會誠實記錄實際用量)。

    延遲 import cv2(經 io.video):讓本模組其餘部分(_percentile 等純函式)
    在沒裝 infer extras 的輕量 venv 也能被匯入與測試。
    """
    from ..io.video import iter_frames

    frames = []
    for _, frame in iter_frames(video_path):
        frames.append(frame)
        if len(frames) >= n_frames:
            break
    return frames


def _sync(device: str | None) -> None:
    if device and str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _run_once(
    frames: list[np.ndarray],
    model_name: str,
    device: str | None,
    quantize: int | str | None,
    warmup: int,
) -> tuple[list[float], list[float]]:
    """單輪:回傳 (每幀純推論延遲, 每幀端到端延遲)(秒),暖身幀已排除。"""
    tracker = PoseTracker(model_name=model_name, device=device)
    kwargs = tracker.track_kwargs()
    if quantize is not None:
        kwargs["quantize"] = quantize

    for i in range(min(warmup, len(frames))):
        tracker.model.track(frames[i], **kwargs)
    _sync(device)

    pure_lat: list[float] = []
    e2e_lat: list[float] = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        results = tracker.model.track(frame, **kwargs)
        _sync(device)
        t1 = time.perf_counter()
        convert_results(i, results)
        t2 = time.perf_counter()
        pure_lat.append(t1 - t0)
        e2e_lat.append(t2 - t0)
    return pure_lat, e2e_lat


def benchmark(
    frames: list[np.ndarray],
    model_name: str,
    device: str | None,
    quantize: int | str | None = None,
    n_runs: int = DEFAULT_N_RUNS,
    warmup: int = DEFAULT_WARMUP,
) -> BenchResult:
    """對一組 (model_name, device, quantize) 跑 n_runs 輪並彙整結果。"""
    pure_fps_runs: list[float] = []
    e2e_fps_runs: list[float] = []
    e2e_lat_pooled: list[float] = []

    for _ in range(n_runs):
        pure_lat, e2e_lat = _run_once(frames, model_name, device, quantize, warmup)
        pure_fps_runs.append(len(pure_lat) / sum(pure_lat))
        e2e_fps_runs.append(len(e2e_lat) / sum(e2e_lat))
        e2e_lat_pooled.extend(e2e_lat)

    return BenchResult(
        model_name=model_name,
        device=str(device) if device else "auto",
        quantize=str(quantize) if quantize is not None else None,
        n_frames=len(frames),
        n_runs=n_runs,
        pure_inference_fps=round(statistics.median(pure_fps_runs), 2),
        end_to_end_fps=round(statistics.median(e2e_fps_runs), 2),
        p50_latency_ms=round(_percentile(e2e_lat_pooled, 0.5) * 1000, 2),
        p95_latency_ms=round(_percentile(e2e_lat_pooled, 0.95) * 1000, 2),
    )
