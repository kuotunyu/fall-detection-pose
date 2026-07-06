"""fall-detection-pose:YOLO26-pose + ByteTrack 的規則式跌倒偵測系統。

核心套件刻意分層:
- ``rules`` / ``events`` / ``eval`` 只依賴 numpy/pandas(無 torch/cv2),可在任何環境秒級測試;
- ``inference`` / ``viz`` / ``app`` 依賴 ultralytics/opencv/gradio(extras:infer、demo)。
"""

__version__ = "0.1.0"
