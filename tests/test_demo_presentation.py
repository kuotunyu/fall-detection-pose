import json

from fall_detection.app.presentation import (
    AnalysisPayload,
    load_analysis_payload,
    load_evidence,
    progress_view,
    render_error_html,
    render_progress_html,
    render_result_html,
    safe_error,
)


def _event(*, rules, track_ids=None, start=1.0, end=2.0):
    return {
        "track_ids": track_ids or [1],
        "start_time_s": start,
        "end_time_s": end,
        "duration_s": end - start,
        "rules_fired": rules,
    }


def test_load_evidence_uses_tracked_results(tmp_path):
    """Wrong benchmark row selection or number formatting must fail this test."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "metrics.json").write_text(
        json.dumps(
            {
                "repository_evidence": {"offline_tests": 123},
                "test_metrics_yolo26n_pose": {"f1": 0.6},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "bench.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "model_name": "yolo26n-pose.pt",
                        "device": "cuda:0",
                        "quantize": None,
                        "end_to_end_fps": 59.65,
                    },
                    {
                        "model_name": "yolo26n-pose.pt",
                        "device": "cuda:0",
                        "quantize": "16",
                        "end_to_end_fps": 64.64,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    items = load_evidence(tmp_path)

    assert [(item.value, item.label, item.detail) for item in items] == [
        ("64.64 FPS", "端到端速度 · T4 FP16", "yolo26n-pose"),
        ("0.600", "Test event-level F1", ""),
        ("123", "離線測試", ""),
    ]


def test_load_evidence_omits_strip_when_evidence_is_missing(tmp_path):
    assert load_evidence(tmp_path) == ()


def test_load_analysis_payload_reads_events_and_metadata(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            {
                "source": "folder/clip.mp4",
                "fps": 30.0,
                "n_events": 0,
                "n_frames": 150,
                "n_tracks": 1,
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    payload = load_analysis_payload(path)

    assert payload.source_name == "clip.mp4"
    assert payload.n_frames == 150
    assert payload.n_tracks == 1
    assert payload.events == ()


def test_result_escapes_source_and_unknown_rule():
    """Removing HTML escaping must expose the malicious fixture and fail."""
    payload = AnalysisPayload(
        source_name='<img src=x onerror="alert(1)">.mp4',
        fps=30.0,
        n_frames=2,
        n_tracks=1,
        events=(_event(rules=["<unsafe>"], track_ids=[1, 7]),),
    )

    result = render_result_html(payload)

    assert "<img" not in result.html
    assert "<unsafe>" not in result.html
    assert "&lt;img" in result.html
    assert "&lt;unsafe&gt;" in result.html


def test_no_event_result_is_explicit_and_has_no_empty_event_grid():
    payload = AnalysisPayload(
        source_name="adl-01.mp4",
        fps=30.0,
        n_frames=150,
        n_tracks=1,
        events=(),
    )

    result = render_result_html(payload)

    assert result.has_events is False
    assert "未偵測到跌倒事件" in result.html
    assert "分析影格" in result.html and ">150<" in result.html
    assert "fd-event-grid" not in result.html
    assert "判定依據" in result.html
    assert "未形成符合條件的事件區間" in result.html
    assert "POSE" in result.html and "EVENT" in result.html


def test_track_lost_path_does_not_claim_alarm_confirmation():
    """A finalized FALLEN track must not be mislabeled as a persisted ALARM."""
    payload = AnalysisPayload(
        source_name="fall.mp4",
        fps=30.0,
        n_frames=100,
        n_tracks=1,
        events=(_event(rules=["track_lost_while_fallen"]),),
    )

    result = render_result_html(payload)

    assert result.state_label == "FALL EVENT"
    assert "Track 消失時已確認 FALLEN" in result.html
    assert "持續躺姿成立" not in result.html


def test_persisted_lying_path_is_labeled_alarm():
    payload = AnalysisPayload(
        source_name="fall.mp4",
        fps=30.0,
        n_frames=100,
        n_tracks=1,
        events=(
            _event(
                rules=["v>v_fall_enter", "posture_vote_confirmed", "lying_persisted"]
            ),
        ),
    )

    result = render_result_html(payload)

    assert result.state_label == "ALARM"
    assert "持續躺姿成立" in result.html


def test_multiple_events_stack_as_separate_articles():
    payload = AnalysisPayload(
        source_name="multi.mp4",
        fps=30.0,
        n_frames=300,
        n_tracks=2,
        events=(
            _event(rules=["lying_persisted"], track_ids=[1]),
            _event(rules=["lying_persisted"], track_ids=[2], start=4.0, end=5.0),
        ),
    )

    result = render_result_html(payload)

    assert result.html.count('class="fd-event"') == 2
    assert "EVENT 1 / 2" in result.html
    assert "EVENT 2 / 2" in result.html


def test_progress_view_maps_pipeline_stage_and_exact_frame_count():
    view = progress_view(0.48, "姿態抽取中 87/150")

    assert view.stage == "POSE"
    assert view.detail == "frame 87 / 150"
    assert view.percent == 48
    assert "87 / 150" in render_progress_html(view)


def test_progress_view_starts_with_video_decode():
    view = progress_view(0.02, "載入模型與影片")

    assert view.stage == "DECODE"
    assert view.title == "影片解碼"


def test_progress_view_clamps_fraction():
    assert progress_view(1.5, "完成").percent == 100
    assert progress_view(-0.2, "準備中").percent == 0


def test_safe_error_does_not_expose_paths_or_traceback():
    err = safe_error(RuntimeError(r"C:\Users\name\secret.mp4 decode failed"))
    html = render_error_html(err)

    assert err.code == "PROCESSING_ERROR"
    assert "C:\\Users" not in err.message
    assert "Traceback" not in err.message
    assert "C:\\Users" not in html


def test_decode_error_has_actionable_copy():
    err = safe_error(ValueError("video decode failed"))

    assert err.code == "VIDEO_DECODE_ERROR"
    assert err.title == "無法讀取這個影片"


def test_result_exposes_pipeline_order():
    payload = AnalysisPayload(
        source_name="fall.mp4",
        fps=30.0,
        n_frames=100,
        n_tracks=1,
        events=(_event(rules=["lying_persisted"]),),
    )

    html = render_result_html(payload).html

    assert "POSE" in html
    assert "TRACK" in html
    assert "RULES" in html
    assert "EVENT" in html
