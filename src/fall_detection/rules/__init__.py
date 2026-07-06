"""規則引擎子套件(純 numpy/pandas,絕不 import torch/ultralytics/cv2)。"""

from .engine import run_engine

__all__ = ["run_engine"]
