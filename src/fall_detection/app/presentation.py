"""Pure presentation helpers for the Gradio demo.

This module intentionally imports neither Gradio nor inference dependencies.  It
turns tracked evidence and ``events.json`` data into escaped, semantic HTML.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvidenceItem:
    value: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class AnalysisPayload:
    source_name: str
    fps: float
    n_frames: int
    n_tracks: int
    events: tuple[dict, ...]


@dataclass(frozen=True)
class ProgressView:
    fraction: float
    percent: int
    stage: str
    title: str
    detail: str


@dataclass(frozen=True)
class SafeError:
    code: str
    title: str
    message: str


@dataclass(frozen=True)
class ResultView:
    html: str
    has_events: bool
    state_label: str


RULE_LABELS = {
    "v>v_fall_enter": "垂直速度超過進入門檻",
    "omega>omega_enter": "軀幹角速度超過進入門檻",
    "posture_vote_confirmed": "躺姿投票確認",
    "lying_persisted": "持續躺姿成立",
    "track_lost_while_fallen": "Track 消失時已確認 FALLEN",
    "track_lost_while_falling_with_lying_posture": "Track 消失時最後姿態符合躺姿",
}


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_evidence(project_root: str | Path, test_count: int = 86) -> tuple[EvidenceItem, ...]:
    """Load the published test F1 and T4 FP16 benchmark from tracked JSON."""

    root = Path(project_root)
    try:
        metrics = json.loads((root / "eval" / "metrics.json").read_text(encoding="utf-8"))
        benchmark = json.loads((root / "bench.json").read_text(encoding="utf-8"))
        fps_row = next(
            row
            for row in benchmark["results"]
            if row["model_name"] == "yolo26n-pose.pt"
            and row["device"] == "cuda:0"
            and str(row.get("quantize")) == "16"
        )
        fps = float(fps_row["end_to_end_fps"])
        f1 = float(metrics["test_metrics_yolo26n_pose"]["f1"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError):
        return ()
    return (
        EvidenceItem(f"{fps:.2f} FPS", "端到端速度 · T4 FP16", "yolo26n-pose"),
        EvidenceItem(f"{f1:.3f}", "Test event-level F1"),
        EvidenceItem(str(int(test_count)), "離線單元測試"),
    )


def load_analysis_payload(path: str | Path) -> AnalysisPayload:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnalysisPayload(
        source_name=Path(str(raw.get("source", ""))).name,
        fps=float(raw.get("fps", 0.0)),
        n_frames=int(raw.get("n_frames", 0)),
        n_tracks=int(raw.get("n_tracks", 0)),
        events=tuple(raw.get("events", [])),
    )


def render_evidence_html(items: tuple[EvidenceItem, ...]) -> str:
    if not items:
        return ""
    cells = "".join(
        (
            '<div class="fd-metric">'
            f'<strong>{_escape(item.value)}</strong>'
            f'<span>{_escape(item.label)}</span>'
            f'<small>{_escape(item.detail)}</small>'
            "</div>"
        )
        for item in items
    )
    return f'<section class="fd-evidence" aria-label="專案量測結果">{cells}</section>'


def progress_view(fraction: float, description: str) -> ProgressView:
    fraction = max(0.0, min(float(fraction), 1.0))
    percent = round(fraction * 100)
    description = str(description)
    frame_match = re.search(r"(\d+)\s*/\s*(\d+)", description)
    detail = f"frame {frame_match.group(1)} / {frame_match.group(2)}" if frame_match else description
    if "姿態" in description or fraction < 0.72:
        stage, title = "POSE", "Pose extraction"
    elif "規則" in description or fraction < 0.82:
        stage, title = "RULES", "Event detection"
    elif "標註" in description or fraction < 1.0:
        stage, title = "ANNOTATE", "Video annotation"
    else:
        stage, title = "COMPLETE", "處理完成"
    return ProgressView(fraction, percent, stage, title, detail)


def render_progress_html(view: ProgressView) -> str:
    stages = (
        ("DECODE", "影片解碼"),
        ("POSE", "Pose extraction"),
        ("RULES", "Event detection"),
        ("ANNOTATE", "Video annotation"),
    )
    order = {name: i for i, (name, _) in enumerate(stages)}
    active_index = order.get(view.stage, len(stages))
    rows = []
    for index, (code, label) in enumerate(stages):
        state = "is-done" if index < active_index else "is-active" if index == active_index else ""
        marker = "✓" if index < active_index else str(index + 1)
        rows.append(
            f'<li class="{state}"><i>{marker}</i><span>{_escape(label)}</span>'
            f'<small>{"RUNNING" if index == active_index else "DONE" if index < active_index else "WAIT"}</small></li>'
        )
    return (
        '<section class="fd-progress" aria-live="polite">'
        '<header><div><small>ANALYSIS PIPELINE</small>'
        f'<h2>{_escape(view.title)}</h2></div><strong>{view.percent}%</strong></header>'
        f'<div class="fd-progress-track"><i style="width:{view.percent}%"></i></div>'
        f'<ol>{"".join(rows)}</ol><footer>{_escape(view.detail)}</footer></section>'
    )


def _event_state(event: dict) -> str:
    rules = set(event.get("rules_fired", []))
    return "ALARM" if "lying_persisted" in rules else "FALL EVENT"


def _render_rule(rule: object) -> str:
    code = str(rule)
    return (
        '<li><div><code>'
        f'{_escape(code)}</code><span>{_escape(RULE_LABELS.get(code, code))}</span>'
        '</div><b>FIRED</b></li>'
    )


def _render_event(event: dict, index: int, total: int) -> str:
    track_ids = ", ".join(str(value) for value in event.get("track_ids", [])) or "—"
    start = float(event.get("start_time_s", 0.0))
    end = float(event.get("end_time_s", 0.0))
    duration = float(event.get("duration_s", end - start))
    state = _event_state(event)
    rules = event.get("rules_fired", [])
    rule_items = "".join(_render_rule(rule) for rule in rules) or '<li class="fd-empty-rule">未記錄觸發規則</li>'
    return (
        '<article class="fd-event">'
        '<header><div><small>事件判定</small>'
        f'<h3>狀態：{_escape(state)}</h3><p>Track ID {_escape(track_ids)}</p></div>'
        f'<b>EVENT {index} / {total}</b></header>'
        '<dl class="fd-event-grid">'
        f'<div><dt>開始</dt><dd>{start:.2f} s</dd></div>'
        f'<div><dt>結束</dt><dd>{end:.2f} s</dd></div>'
        f'<div><dt>時長</dt><dd>{duration:.2f} s</dd></div>'
        f'<div><dt>Track ID</dt><dd>{_escape(track_ids)}</dd></div>'
        "</dl>"
        f'<section class="fd-rules"><h4>Rules fired</h4><ul>{rule_items}</ul></section>'
        "</article>"
    )


def render_result_html(payload: AnalysisPayload) -> ResultView:
    source = _escape(payload.source_name or "未命名影片")
    if not payload.events:
        body = (
            '<section class="fd-no-event">'
            '<header><span aria-hidden="true">✓</span><div><small>EVENT RESULT</small>'
            '<h2>未偵測到跌倒事件</h2></div></header>'
            '<p>完整影片未產生符合確認條件的 ALARM 狀態。</p>'
            '<dl>'
            f'<div><dt>分析影格</dt><dd>{payload.n_frames}</dd></div>'
            f'<div><dt>追蹤人物</dt><dd>{payload.n_tracks}</dd></div>'
            '<div><dt>跌倒事件</dt><dd>0</dd></div>'
            "</dl></section>"
        )
        return ResultView(body, False, "NO EVENT")
    events = "".join(
        _render_event(event, index, len(payload.events))
        for index, event in enumerate(payload.events, start=1)
    )
    state = _event_state(payload.events[0])
    body = (
        '<section class="fd-result-panel">'
        f'<div class="fd-source"><span>{source}</span><span>{payload.fps:g} FPS</span></div>'
        f'<div class="fd-events">{events}</div></section>'
    )
    return ResultView(body, True, state)


def safe_error(exc: Exception) -> SafeError:
    raw = str(exc)
    lower = raw.lower()
    if isinstance(exc, ValueError) and (
        "decode" in lower or "codec" in lower or "解碼" in raw
    ):
        return SafeError(
            "VIDEO_DECODE_ERROR",
            "無法讀取這個影片",
            "檔案可能已損毀，或使用目前不支援的編碼格式。請改用 MP4 或 MOV 短片。",
        )
    if isinstance(exc, ValueError) and ("上傳" in raw or "video" in lower):
        return SafeError("INPUT_REQUIRED", "請先選擇影片", "選擇一個短片後再開始分析。")
    return SafeError(
        "PROCESSING_ERROR",
        "影片處理未完成",
        "系統未能完成這次分析。請重新執行，或改用另一個短片。",
    )


def render_error_html(error: SafeError) -> str:
    return (
        '<section class="fd-error" role="alert">'
        f'<small>{_escape(error.code)}</small><h2>{_escape(error.title)}</h2>'
        f'<p>{_escape(error.message)}</p>'
        '<details><summary>診斷資訊</summary><p>若問題持續發生，請保留錯誤代碼並檢查影片格式。</p></details>'
        "</section>"
    )
