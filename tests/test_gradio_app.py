"""app.gradio_app 的純函式測試(不碰 gradio/torch/ultralytics/cv2)。

``build_demo``/``process_video`` 需要 gradio + infer extras 才能真正執行,只能在
Colab 驗證;這裡只測試不依賴任何重依賴的部分:events dict → gr.Dataframe rows
的轉換。模組頂層不 import gradio,所以這個 import 本身就是一項回歸測試——
只要 gradio_app.py 不小心在頂層引了重依賴,這裡就會在本機輕量 venv 炸掉。
"""

from fall_detection.app.gradio_app import EVENT_TABLE_HEADERS, _events_to_rows


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
