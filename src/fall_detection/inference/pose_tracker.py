"""YOLO26-pose + ByteTrack 的薄封裝。

設計要點(對應 Ultralytics 官方文件確認過的行為):
- ``persist=True`` 僅用於「自己逐幀餵」的迴圈(本模組正是),
  讓 tracker 狀態跨幀延續;
- ``results[0].boxes.id`` 可能為 None(該幀無已確認 track)→ 哨兵 -1;
- ``results[0].keypoints.conf`` 可能為 None → 哨兵 -1.0(下游一律視為不可信);
- 換影片前必須 reset,否則 track id 與 tracker 狀態會跨影片汙染。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.cache import N_KPTS


@dataclass
class FrameDetections:
    """單幀所有人的偵測結果(numpy,已脫離 torch)。"""

    frame_idx: int
    boxes: np.ndarray  # (N, 4) xyxy
    box_conf: np.ndarray  # (N,)
    track_ids: np.ndarray  # (N,) int32;-1 = 未指派 track
    kpts_xy: np.ndarray  # (N, 17, 2)
    kpts_conf: np.ndarray  # (N, 17);-1.0 = 模型未輸出 conf

    @property
    def n(self) -> int:
        return len(self.boxes)


def _empty(frame_idx: int) -> FrameDetections:
    return FrameDetections(
        frame_idx=frame_idx,
        boxes=np.zeros((0, 4), dtype=np.float32),
        box_conf=np.zeros((0,), dtype=np.float32),
        track_ids=np.zeros((0,), dtype=np.int32),
        kpts_xy=np.zeros((0, N_KPTS, 2), dtype=np.float32),
        kpts_conf=np.zeros((0, N_KPTS), dtype=np.float32),
    )


def convert_results(frame_idx: int, results) -> FrameDetections:
    """單幀的 ``model.track()`` 原始回傳(torch tensors)→ 純 numpy FrameDetections。

    獨立成函式(而非 PoseTracker 的方法)供 ``bench.benchmark`` 復用:量測
    「純推論」與「端到端」延遲時,兩者都只呼叫一次 ``model.track()``,轉換
    這步驟另外計時,而不是把轉換邏輯複製一份。
    """
    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return _empty(frame_idx)
    n = len(boxes)

    ids = boxes.id
    track_ids = (
        ids.int().cpu().numpy().astype(np.int32)
        if ids is not None
        else np.full((n,), -1, dtype=np.int32)
    )

    kpts = r.keypoints
    if kpts is None or kpts.xy is None:
        kxy = np.zeros((n, N_KPTS, 2), dtype=np.float32)
        kconf = np.full((n, N_KPTS), -1.0, dtype=np.float32)
    else:
        kxy = kpts.xy.cpu().numpy().astype(np.float32)
        kconf = (
            kpts.conf.cpu().numpy().astype(np.float32)
            if kpts.conf is not None
            else np.full((n, N_KPTS), -1.0, dtype=np.float32)
        )

    return FrameDetections(
        frame_idx=frame_idx,
        boxes=boxes.xyxy.cpu().numpy().astype(np.float32),
        box_conf=boxes.conf.cpu().numpy().astype(np.float32),
        track_ids=track_ids,
        kpts_xy=kxy,
        kpts_conf=kconf,
    )


class PoseTracker:
    def __init__(
        self,
        model_name: str,
        tracker_yaml: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.model_name = model_name
        self.tracker_yaml = tracker_yaml
        self.conf = conf
        self.iou = iou
        self.device = device

    def track_kwargs(self) -> dict:
        """組出 ``model.track()`` 的關鍵字參數(benchmark 需要直接呼叫底層
        ``model.track()`` 以量測純推論延遲,不能只靠 :meth:`track_frame`,
        避免同一幀被 ``persist=True`` 的 tracker 吃兩次而弄亂 track 狀態)。"""
        return dict(
            persist=True,
            tracker=self.tracker_yaml,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

    def track_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameDetections:
        """對單一幀執行 pose 推論 + 追蹤;回傳純 numpy 結果。"""
        results = self.model.track(frame_bgr, **self.track_kwargs())
        return convert_results(frame_idx, results)

    def reset(self) -> None:
        """清空 tracker 狀態(換影片前呼叫,避免 track id 跨影片延續)。"""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        if trackers:
            for t in trackers:
                t.reset()
        # 若 ultralytics 內部結構改版導致上面拿不到 tracker,
        # 重載模型是保底做法(慢但絕對乾淨)
        elif predictor is not None:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)

    @staticmethod
    def ultralytics_version() -> str:
        import ultralytics

        return ultralytics.__version__
