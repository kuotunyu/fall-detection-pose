"""Opt-in end-to-end smoke test for the real Ultralytics inference stack."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_INFERENCE_SMOKE = os.environ.get("FDP_RUN_INFERENCE_SMOKE") == "1"


@pytest.mark.inference
@pytest.mark.skipif(
    not RUN_INFERENCE_SMOKE,
    reason="set FDP_RUN_INFERENCE_SMOKE=1 to run the real model pipeline",
)
def test_real_model_pipeline_writes_machine_readable_and_video_outputs(tmp_path):
    """Breaking model, tracker, rules, CLI, or annotation wiring must fail."""
    source = REPO_ROOT / "assets" / "demo_fall.gif"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fall_detection.cli",
            "pipeline",
            "--source",
            str(source),
            "--out-dir",
            str(tmp_path),
            "--config",
            str(REPO_ROOT / "config.yaml"),
            "--model",
            "yolo26n-pose.pt",
            "--device",
            "cpu",
            "--debug",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr

    events_path = tmp_path / "demo_fall.events.json"
    cache_path = tmp_path / "demo_fall.parquet"
    debug_path = tmp_path / "demo_fall.debug.jsonl"
    video_path = tmp_path / "demo_fall_annotated.mp4"
    assert events_path.is_file()
    assert cache_path.is_file()
    assert debug_path.is_file()
    assert video_path.is_file()

    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert len(events["events"]) == 1
    event = events["events"][0]
    assert event["track_ids"]
    assert all(track_id > 0 for track_id in event["track_ids"])
    assert "lying_persisted" in event["rules_fired"]
    assert event["duration_s"] > 0

    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened()
        ok, frame = capture.read()
        assert ok
        assert frame.size > 0
    finally:
        capture.release()
