"""bench.benchmark 的純函式測試(不碰 torch/ultralytics/cv2)。

``PoseTracker``/``load_frames`` 需要 infer extras 才能真正執行,只能在 Colab
(GPU runtime)驗證;這裡只測試不依賴任何模型/影片的部分:百分位數計算、
BenchResult 的序列化形狀。
"""

from fall_detection.bench.benchmark import BenchResult, _percentile


def test_percentile_median_of_five():
    assert _percentile([1, 2, 3, 4, 5], 0.5) == 3.0


def test_percentile_p95_interpolates():
    assert _percentile([1, 2, 3, 4, 5], 0.95) == 4.8


def test_percentile_single_value():
    assert _percentile([10.0], 0.5) == 10.0
    assert _percentile([10.0], 0.95) == 10.0


def test_percentile_unsorted_input():
    assert _percentile([5, 1, 3, 2, 4], 0.5) == 3.0


def test_bench_result_to_dict_roundtrip():
    r = BenchResult(
        model_name="yolo26n-pose.pt",
        device="cuda:0",
        quantize=None,
        n_frames=300,
        n_runs=3,
        pure_inference_fps=45.0,
        end_to_end_fps=42.0,
        p50_latency_ms=22.0,
        p95_latency_ms=30.0,
    )
    d = r.to_dict()
    assert d["model_name"] == "yolo26n-pose.pt"
    assert d["quantize"] is None
    assert d["n_frames"] == 300
