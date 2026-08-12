"""app.gradio_app 的純函式測試(不碰 gradio/torch/ultralytics/cv2)。

``build_demo``/``process_video`` 需要 gradio、infer extras、模型權重與影片資源；
這裡聚焦不依賴重型套件的部分：events dict → gr.Dataframe rows 的轉換。
模組頂層不 import gradio，因此這個 import 本身就是一項回歸測試——只要
gradio_app.py 不小心在頂層引入重型依賴，這裡就會在輕量 venv 失敗。
"""

from types import SimpleNamespace

import pandas as pd

from fall_detection.app.gradio_app import (
    EVENT_TABLE_HEADERS,
    _analysis_metadata,
    _events_to_rows,
    _stream_process,
)
from fall_detection.app.theme import APP_HEADER_HTML


def _event(track_ids, start, end, rules=None):
    return {
        "track_ids": track_ids,
        "start_time_s": start,
        "end_time_s": end,
        "duration_s": round(end - start, 3),
        "rules_fired": rules or [],
    }


def test_events_to_rows_empty():
    assert _events_to_rows([]) == []


def test_events_to_rows_formats_fields():
    events = [_event([1, 7], 0.933, 2.233, ["track_lost_while_fallen"])]
    rows = _events_to_rows(events)
    assert rows == [["1,7", 0.93, 2.23, 1.3, "track_lost_while_fallen"]]


def test_events_to_rows_multiple_rules_joined():
    events = [_event([3], 1.0, 1.5, ["a", "b"])]
    rows = _events_to_rows(events)
    assert rows[0][4] == "a, b"


def test_events_to_rows_multiple_track_ids_comma_joined():
    events = [_event([2, 9, 14], 0.0, 1.0)]
    rows = _events_to_rows(events)
    assert rows[0][0] == "2,9,14"


def test_event_table_headers_length_matches_row_length():
    events = [_event([1], 0.0, 1.0)]
    rows = _events_to_rows(events)
    assert len(EVENT_TABLE_HEADERS) == len(rows[0])


def test_events_to_rows_preserves_order():
    events = [_event([1], 0.0, 1.0), _event([2], 5.0, 6.0)]
    rows = _events_to_rows(events)
    assert [r[0] for r in rows] == ["1", "2"]


def test_analysis_metadata_uses_video_frame_count_and_valid_track_ids():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 0, 1, 1, 2],
            "track_id": [-1, 4, 4, 9, -1],
        }
    )

    metadata = _analysis_metadata(df, SimpleNamespace(n_frames=150))

    assert metadata == {"n_frames": 150, "n_tracks": 2}


def test_analysis_metadata_handles_empty_detections():
    df = pd.DataFrame({"frame_idx": [], "track_id": []})

    metadata = _analysis_metadata(df, SimpleNamespace(n_frames=24))

    assert metadata == {"n_frames": 24, "n_tracks": 0}


def test_stream_process_yields_progress_then_success():
    def runner(video_path, model_name, config_path, on_progress):
        assert (video_path, model_name, config_path) == (
            "clip.mp4",
            "yolo26n-pose.pt",
            "config.yaml",
        )
        on_progress(0.25, "分析影格 4/16")
        on_progress(0.75, "事件規則運算")
        return "annotated.mp4", [], "events.json"

    messages = list(
        _stream_process("clip.mp4", "yolo26n-pose.pt", "config.yaml", runner=runner)
    )

    assert [(message.kind, message.fraction) for message in messages] == [
        ("progress", 0.25),
        ("progress", 0.75),
        ("success", 1.0),
    ]
    assert messages[-1].annotated_path == "annotated.mp4"
    assert messages[-1].events_path == "events.json"


def test_stream_process_yields_terminal_error_without_raising():
    failure = RuntimeError("private implementation detail")

    def runner(*_args, **_kwargs):
        raise failure

    messages = list(
        _stream_process("clip.mp4", "yolo26n-pose.pt", "config.yaml", runner=runner)
    )

    assert len(messages) == 1
    assert messages[0].kind == "error"
    assert messages[0].error is failure


def test_header_links_to_public_repository():
    assert "https://github.com/kuotunyu/fall-detection-pose" in APP_HEADER_HTML
