# Gradio Demo UI/UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default Gradio demo with the approved Traditional Chinese scientific dashboard, complete UI states, responsive event presentation, and refreshed README media.

**Architecture:** Keep the existing extract → rules → annotate pipeline and its legacy `process_video()` tuple stable. Add a pure presentation module for evidence loading, HTML escaping/rendering, error classification, and progress messages; use Gradio only inside `build_demo()`. The Gradio handler streams progress through a background worker queue, then switches one of four mutually exclusive UI states: input, processing, result, or error.

**Tech Stack:** Python 3.10+, Gradio 6.x, pytest, Pydantic/YAML project configuration, HTML/CSS, Playwright, FFmpeg.

## Global Constraints

- Traditional Chinese (`zh-TW`) is the primary language; preserve conventional terms such as `YOLO26-pose`, `ByteTrack`, `Track ID`, `ALARM`, `F1`, `FPS`, and `Rules fired`.
- Use the approved palette: `#DEDFD9`, `#EEEBE4`, `#F9F8F4`, `#24312D`, `#5C6964`, `#687F74`, `#E7EBE5`, `#C66C5D`, `#F0DED9`.
- Structural regions use 0–4 px corner radius; no gradients, decorative pills, or nested card shadows.
- Desktop rule conditions are 18 px; event-summary values are 23 px. Mobile body text and rule conditions are at least 16 px, and touch controls are at least 48 px high.
- The production video keeps its original aspect ratio and must not crop or obscure annotations.
- Dynamic filenames, errors, rule values, and event fields are HTML-escaped.
- The UI consumes the existing pipeline output and must not introduce a second fall-detection rule path.
- The previous test GitHub account must not appear in source, UI copy, README, notebooks, or configuration.
- Do not change pose extraction, tracking, fall rules, evaluation protocol, or benchmark methodology.

## File Structure

- Create `src/fall_detection/app/presentation.py`: pure dataclasses, evidence loading, event/progress/error view models, and escaped HTML renderers. This module imports no Gradio or inference dependencies.
- Create `src/fall_detection/app/theme.py`: approved CSS and static application-header HTML.
- Modify `src/fall_detection/app/gradio_app.py`: preserve pipeline behavior, write analysis metadata, stream progress, and compose four Gradio UI states.
- Replace `tests/test_gradio_app.py`: retain legacy row-format coverage and add pipeline-metadata/progress-stream tests.
- Create `tests/test_demo_presentation.py`: pure presentation/evidence/error/escaping tests.
- Create `tests/test_gradio_build.py`: optional Gradio construction smoke test guarded by `pytest.importorskip("gradio")`.
- Create `scripts/capture_demo_media.py`: Playwright capture of final desktop/mobile states and README media.
- Modify `README.md`: updated demo media, captions, and local-demo guidance.
- Modify `notebooks/05_gradio_demo.ipynb`: update visible instructions and example sequence from the stale `fall-01` wording to the recorded project evidence.
- Update `assets/demo_fall.gif` and create `assets/demo_adl.png`; optionally create `assets/demo_mobile.png` only if legible at GitHub width.

---

### Task 1: Pure presentation model and evidence rendering

**Files:**
- Create: `src/fall_detection/app/presentation.py`
- Create: `src/fall_detection/app/theme.py`
- Create: `tests/test_demo_presentation.py`

**Interfaces:**
- Produces: `EvidenceItem`, `AnalysisPayload`, `ProgressView`, `SafeError`, `load_evidence()`, `load_analysis_payload()`, `progress_view()`, `safe_error()`, `render_evidence_html()`, `render_progress_html()`, `render_result_html()`, `render_error_html()`, `DEMO_CSS`, and `APP_HEADER_HTML`.
- Consumes: tracked `eval/metrics.json`, `bench.json`, and the existing `events.json` schema.

- [ ] **Step 1: Write failing evidence and payload tests**

```python
import json

from fall_detection.app.presentation import load_analysis_payload, load_evidence


def test_load_evidence_uses_tracked_results(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "metrics.json").write_text(
        json.dumps({"test_metrics_yolo26n_pose": {"f1": 0.6}}), encoding="utf-8"
    )
    (tmp_path / "bench.json").write_text(
        json.dumps({"results": [{"model_name": "yolo26n-pose.pt", "device": "cuda:0",
                                  "quantize": "16", "end_to_end_fps": 64.64}]}),
        encoding="utf-8",
    )
    items = load_evidence(tmp_path, test_count=86)
    assert [(item.value, item.label) for item in items] == [
        ("64.64 FPS", "端到端速度 · T4 FP16"),
        ("0.600", "Test event-level F1"),
        ("86", "離線單元測試"),
    ]


def test_load_analysis_payload_reads_events_and_metadata(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"source": "clip.mp4", "fps": 30.0, "n_events": 0,
                                "n_frames": 150, "n_tracks": 1, "events": []}),
                    encoding="utf-8")
    payload = load_analysis_payload(path)
    assert payload.source_name == "clip.mp4"
    assert payload.n_frames == 150
    assert payload.n_tracks == 1
    assert payload.events == ()
```

- [ ] **Step 2: Run the new tests and verify the missing-module failure**

Run: `uv run pytest tests/test_demo_presentation.py -v`

Expected: collection fails because `fall_detection.app.presentation` does not exist.

- [ ] **Step 3: Implement dataclasses and JSON loaders**

```python
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


def load_analysis_payload(path: str | Path) -> AnalysisPayload:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnalysisPayload(
        source_name=Path(str(raw.get("source", ""))).name,
        fps=float(raw.get("fps", 0.0)),
        n_frames=int(raw.get("n_frames", 0)),
        n_tracks=int(raw.get("n_tracks", 0)),
        events=tuple(raw.get("events", [])),
    )
```

`load_evidence()` must select the `yolo26n-pose.pt` / `cuda:0` / `quantize == "16"` benchmark row and format F1 with three decimals. Return an empty tuple if either evidence file is missing or malformed.

- [ ] **Step 4: Add failing HTML-escaping, rule-path, empty-result, progress, and error tests**

```python
from fall_detection.app.presentation import (
    AnalysisPayload, progress_view, render_result_html, safe_error,
)


def test_result_escapes_source_and_unknown_rule():
    payload = AnalysisPayload(
        source_name='<img src=x onerror="alert(1)">.mp4', fps=30, n_frames=2, n_tracks=1,
        events=({"track_ids": [1], "start_time_s": 1.0, "end_time_s": 2.0,
                 "duration_s": 1.0, "rules_fired": ["<unsafe>"]},),
    )
    html = render_result_html(payload)
    assert "<img" not in html and "<unsafe>" not in html
    assert "&lt;img" in html
    assert "&lt;unsafe&gt;" in html


def test_no_event_result_is_explicit_and_has_no_empty_table():
    payload = AnalysisPayload("adl-01.mp4", 30, 150, 1, ())
    html = render_result_html(payload)
    assert "未偵測到跌倒事件" in html
    assert "分析影格" in html and ">150<" in html
    assert "event-grid" not in html


def test_track_lost_path_does_not_claim_lying_persisted():
    payload = AnalysisPayload(
        "fall.mp4", 30, 100, 1,
        ({"track_ids": [1], "start_time_s": 1.0, "end_time_s": 1.6,
          "duration_s": 0.6, "rules_fired": ["track_lost_while_fallen"]},),
    )
    html = render_result_html(payload)
    assert "Track 消失時已確認 FALLEN" in html
    assert "持續躺姿成立" not in html


def test_progress_view_maps_pipeline_stage_and_exact_frame_count():
    view = progress_view(0.48, "姿態抽取中 87/150")
    assert view.stage == "POSE"
    assert view.detail == "frame 87 / 150"
    assert view.percent == 48


def test_safe_error_does_not_expose_paths_or_traceback():
    err = safe_error(RuntimeError(r"C:\\Users\\name\\secret.mp4 decode failed"))
    assert err.code == "PROCESSING_ERROR"
    assert "C:\\Users" not in err.message
    assert "Traceback" not in err.message
```

- [ ] **Step 5: Implement escaped renderers and error classification**

Use `html.escape(..., quote=True)` for every dynamic value. Map known rules exactly:

```python
RULE_LABELS = {
    "v>v_fall_enter": "垂直速度超過進入門檻",
    "omega>omega_enter": "軀幹角速度超過進入門檻",
    "posture_vote_confirmed": "躺姿投票確認",
    "lying_persisted": "持續躺姿成立",
    "track_lost_while_fallen": "Track 消失時已確認 FALLEN",
    "track_lost_while_falling_with_lying_posture": "Track 消失時最後姿態符合躺姿",
}
```

Unknown rules render their escaped original code. `render_result_html()` produces a fall state when events exist and the explicit zero-event state otherwise. Multiple events stack as semantic `<article>` elements; mobile layout is controlled by CSS, not duplicate data.

- [ ] **Step 6: Implement the approved theme and header**

Create `DEMO_CSS` using the exact global palette and type/radius constraints. Required selectors include `.fd-shell`, `.fd-header`, `.fd-evidence`, `.fd-workspace`, `.fd-result-panel`, `.fd-rules`, `.fd-event-grid`, `.fd-progress`, `.fd-error`, and the `<650px` responsive rules. Set Gradio root width to use the browser canvas and override component containers without removing focus outlines.

- [ ] **Step 7: Run pure presentation tests**

Run: `uv run pytest tests/test_demo_presentation.py -v`

Expected: all tests pass.

- [ ] **Step 8: Commit the presentation layer**

```bash
git add src/fall_detection/app/presentation.py src/fall_detection/app/theme.py tests/test_demo_presentation.py
git commit -m "Add demo presentation layer"
```

---

### Task 2: Preserve pipeline compatibility and expose metadata/progress

**Files:**
- Modify: `src/fall_detection/app/gradio_app.py`
- Modify: `tests/test_gradio_app.py`

**Interfaces:**
- Consumes: `AnalysisPayload` loader from Task 1 and existing `process_video()` callers.
- Produces: unchanged `process_video(...) -> tuple[str, list[list], str]`, metadata inside `events.json`, `StreamMessage`, and `_stream_process(...) -> Iterator[StreamMessage]`.

- [ ] **Step 1: Add failing metadata and stream tests**

```python
from fall_detection.app.gradio_app import _stream_process


def test_stream_process_emits_progress_then_success(tmp_path):
    events_path = tmp_path / "events.json"
    events_path.write_text('{"source":"x.mp4","fps":30,"n_events":0,"events":[]}',
                           encoding="utf-8")

    def fake_runner(video, model, config, on_progress):
        on_progress(0.5, "姿態抽取中 3/6")
        return "annotated.mp4", [], str(events_path)

    messages = list(_stream_process("x.mp4", "yolo26n-pose.pt", "config.yaml",
                                    runner=fake_runner))
    assert [message.kind for message in messages] == ["progress", "success"]
    assert messages[0].description == "姿態抽取中 3/6"
    assert messages[-1].annotated_path == "annotated.mp4"


def test_stream_process_converts_exception_to_error_message():
    def failing_runner(video, model, config, on_progress):
        raise RuntimeError("boom")

    messages = list(_stream_process("x.mp4", "", "config.yaml", runner=failing_runner))
    assert len(messages) == 1
    assert messages[0].kind == "error"
    assert isinstance(messages[0].error, RuntimeError)
```

- [ ] **Step 2: Run stream tests and confirm failure**

Run: `uv run pytest tests/test_gradio_app.py -v`

Expected: fail because `_stream_process` is missing.

- [ ] **Step 3: Add metadata to the existing JSON without changing tuple output**

Immediately before `write_events_json()`, derive:

```python
n_frames = int(df["frame_idx"].nunique()) if not df.empty else 0
n_tracks = int(df["track_id"].nunique()) if not df.empty else 0
write_events_json(
    events_path,
    events,
    source=str(video_path),
    fps=meta.fps,
    extra={"n_frames": n_frames, "n_tracks": n_tracks},
)
```

Keep `_events_to_rows()` and the return tuple unchanged for notebook/backward compatibility.

- [ ] **Step 4: Implement the queue-backed progress stream**

Define frozen `StreamMessage` fields: `kind`, `fraction`, `description`, `annotated_path`, `events_path`, and `error`. `_stream_process()` launches one daemon `threading.Thread`; the callback puts progress messages into `queue.Queue`, and the worker puts exactly one terminal success/error message. The iterator blocks on `queue.get()` and stops after the terminal message, avoiding polling sleeps and thread leaks.

- [ ] **Step 5: Run lightweight regression tests**

Run: `uv run pytest tests/test_gradio_app.py tests/test_state_machine.py -v`

Expected: all tests pass; the module still imports without Gradio or inference extras at import time.

- [ ] **Step 6: Commit pipeline metadata and progress streaming**

```bash
git add src/fall_detection/app/gradio_app.py tests/test_gradio_app.py
git commit -m "Expose demo progress and result metadata"
```

---

### Task 3: Compose the Gradio scientific dashboard and four UI states

**Files:**
- Modify: `src/fall_detection/app/gradio_app.py`
- Create: `tests/test_gradio_build.py`
- Modify: `tests/test_demo_presentation.py`

**Interfaces:**
- Consumes: `DEMO_CSS`, `APP_HEADER_HTML`, presentation renderers, and `_stream_process()`.
- Produces: `build_demo(config_path="config.yaml", example_videos=None)` with input, processing, result, and error groups; `gr.Blocks` remains the public interface.

- [ ] **Step 1: Add a guarded Gradio construction test**

```python
import pytest

gr = pytest.importorskip("gradio")

from fall_detection.app.gradio_app import build_demo


def test_build_demo_contains_named_state_groups():
    demo = build_demo()
    config = demo.get_config_file()
    ids = {component.get("props", {}).get("elem_id") for component in config["components"]}
    assert {"fd-input", "fd-processing", "fd-result", "fd-error"} <= ids
```

- [ ] **Step 2: Run the build test and verify it fails**

Run: `uv run pytest tests/test_gradio_build.py -v`

Expected: fail because the named state groups do not exist.

- [ ] **Step 3: Replace the default two-column Gradio layout**

Build `gr.Blocks(title="姿態追蹤式跌倒事件偵測", css=DEMO_CSS, fill_width=True)` with:

- Static header/method/evidence HTML.
- `fd-input` visible initially: `gr.Video`, compact examples, a closed `gr.Accordion("進階設定")` containing model dropdown, and `開始分析`.
- `fd-processing` hidden initially: progress HTML only; no blank result video/table/file.
- `fd-result` hidden initially: annotated `gr.Video`, rendered semantic result HTML, replacement `gr.Video`, `重新執行`, and `events.json` file.
- `fd-error` hidden initially: safe error HTML, replacement input, and `重新嘗試`.

Do not render `gr.Dataframe`.

- [ ] **Step 4: Wire the streaming handler and visibility updates**

The Gradio generator translates stream messages into eight outputs:

```python
[input_group, processing_group, result_group, error_group,
 progress_html, video_out, result_html, file_out]
```

For progress: hide input/result/error, show processing, and update `render_progress_html(progress_view(...))`.

For success: load the JSON payload; hide input/processing/error; show result; set annotated video, `render_result_html(payload)`, and JSON file.

For error: hide input/processing/result; show error; set `render_error_html(safe_error(exc))`; clear output/file values.

Bind the same handler factory to the initial, replacement, and error-retry inputs. Set `concurrency_limit=1`, `show_progress="hidden"`, and `scroll_to_output=True` so the custom state is the sole progress UI.

- [ ] **Step 5: Test real rule-path rendering and state invariants**

Add assertions that a `track_lost_while_fallen` event does not display `ALARM` as if `lying_persisted` fired; label it as a finalized fall event and show the actual rule path. Confirm waiting HTML contains no blank output labels.

- [ ] **Step 6: Run component and presentation tests**

Run: `uv run pytest tests/test_demo_presentation.py tests/test_gradio_app.py tests/test_gradio_build.py -v`

Expected: all tests pass (or the build test skips only when the demo extra is absent).

- [ ] **Step 7: Run the complete lightweight suite and Ruff**

Run: `uv run pytest -q && uv run ruff check .`

Expected: all existing tests pass and Ruff reports no errors.

- [ ] **Step 8: Commit the dashboard**

```bash
git add src/fall_detection/app/gradio_app.py tests/test_gradio_build.py tests/test_demo_presentation.py
git commit -m "Redesign the Gradio demo dashboard"
```

---

### Task 4: Browser verification and responsive fixes

**Files:**
- Modify: `src/fall_detection/app/theme.py`
- Modify: `src/fall_detection/app/presentation.py`
- Create: `scripts/capture_demo_media.py`

**Interfaces:**
- Consumes: local Gradio app and state-specific component IDs.
- Produces: repeatable Playwright screenshots and overflow/console assertions.

- [ ] **Step 1: Add a capture script with browser assertions**

The script accepts `--url`, `--video`, `--out-dir`, and `--mobile`. It must:

```python
page.goto(args.url, wait_until="networkidle")
assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
assert not console_errors
page.screenshot(path=out_dir / name, full_page=True)
```

It uploads the requested video, clicks `開始分析`, waits for either `#fd-result` or `#fd-error`, and raises on an error state. Mobile uses a 390 × 844 viewport; desktop captures both 1440 × 1000 and 1800 × 1000.

- [ ] **Step 2: Start the local app using the test helper**

Run `with_server.py --help` first, then run the app on an unused port with:

```powershell
uv run python C:\Users\3Hml\.codex\skills\webapp-testing\scripts\with_server.py `
  --server "uv run python -m fall_detection.app.gradio_app --no-share --server-port 7863" `
  --port 7863 -- uv run --with playwright python scripts/capture_demo_media.py `
  --url http://127.0.0.1:7863 --out-dir outputs/ui-check
```

- [ ] **Step 3: Inspect desktop and mobile captures**

Use the local image viewer on each PNG. Specifically verify rule conditions, event values, input instructions, progress rows, and the no-event conclusion at normal zoom.

- [ ] **Step 4: Fix CSS one issue at a time and rerun the capture**

Permitted fixes are restricted to overflow, font size, spacing, state visibility, focus visibility, and video aspect ratio. Do not change the approved palette, scientific wording, low-radius geometry, or inference behavior.

- [ ] **Step 5: Run automated regressions after visual fixes**

Run: `uv run pytest -q && uv run ruff check .`

Expected: all tests and Ruff pass.

- [ ] **Step 6: Commit verified responsive behavior**

```bash
git add src/fall_detection/app/theme.py src/fall_detection/app/presentation.py scripts/capture_demo_media.py
git commit -m "Verify responsive demo states"
```

---

### Task 5: Generate actual-pipeline README media and update documentation

**Files:**
- Modify: `README.md`
- Modify: `notebooks/05_gradio_demo.ipynb`
- Update: `assets/demo_fall.gif`
- Create: `assets/demo_adl.png`
- Optional create: `assets/demo_mobile.png`

**Interfaces:**
- Consumes: final local Gradio interface, official URFD `fall-06` and `adl-01` sequences, existing pipeline, and `scripts/capture_demo_media.py`.
- Produces: GitHub-ready media and accurate reproduction guidance.

- [ ] **Step 1: Install locked demo and inference dependencies**

Run: `uv sync --locked --extra infer --extra demo`

Expected: Gradio 6 and inference dependencies install successfully from the lockfile.

- [ ] **Step 2: Download only the two required official URFD sequences into ignored data**

Run:

```powershell
uv run python -c "from fall_detection.io import urfd; d='data/urfd_demo'; s=['fall-06','adl-01']; urfd.download_sequences(d,s); urfd.build_videos(d,s)"
```

Expected: `data/urfd_demo/videos/fall-06.mp4` and `adl-01.mp4` exist; no dataset file is staged.

- [ ] **Step 3: Record the final fall flow**

Use Playwright video recording while uploading `fall-06.mp4`, starting analysis, and waiting for the result. Capture the initial state for about one second, processing state until completion, and result for two seconds. Convert the recording to a 12–15 FPS GIF with FFmpeg palette generation; cap width at 1280 px and keep the result under 5 MB.

Expected output: `assets/demo_fall.gif`, showing the redesigned UI around an actual pipeline result.

- [ ] **Step 4: Capture ADL and mobile screenshots**

Upload `adl-01.mp4`, wait for the explicit `未偵測到跌倒事件` state, and save `assets/demo_adl.png`. Save `assets/demo_mobile.png` only if its text remains readable when rendered at 390 px; otherwise omit it.

- [ ] **Step 5: Update README demo presentation**

Replace the old vertical GIF table with:

- One-sentence description of pose tracking and rule-based event output.
- Fall GIF captioned as actual `fall-06` pipeline output.
- ADL/no-event screenshot captioned as actual `adl-01` pipeline output.
- Existing measured evidence table immediately after the media.
- Local demo command as the primary interactive path.
- Colab notebooks retained under reproduction, not described as the primary demo experience.
- Existing URFD attribution and CC BY-NC-SA notice retained adjacent to media.

- [ ] **Step 6: Update notebook instructions mechanically**

Change the visible `05_gradio_demo.ipynb` instructions and `example_candidates` to match the README evidence (`fall-06`, `adl-01`), the redesigned state names, and current output order. Use a notebook-aware formatter/script; do not hand-edit JSON formatting.

- [ ] **Step 7: Verify documentation and media**

Run:

```powershell
Get-Item assets\demo_fall.gif,assets\demo_adl.png | Select-Object Name,Length
rg -n -i "kuotunyu|fall-01.*ALARM|Colab.*primary" README.md notebooks src tests
uv run pytest -q
uv run ruff check .
git diff --check
```

Expected: GIF is under 5 MB, old-account search is empty, stale fall-demo wording is absent, all tests pass, Ruff is clean, and patch whitespace is clean.

- [ ] **Step 8: Commit README media and guidance**

```bash
git add README.md notebooks/05_gradio_demo.ipynb assets/demo_fall.gif assets/demo_adl.png
git add assets/demo_mobile.png  # only when created and legible
git commit -m "Refresh demo media and README"
```

---

### Task 6: Final end-to-end verification

**Files:**
- Verify only; modify the smallest responsible file if a check fails.

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: a clean, tested branch ready to integrate.

- [ ] **Step 1: Run the complete automated suite**

Run: `uv run pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run static analysis**

Run: `uv run ruff check .`

Expected: zero errors.

- [ ] **Step 3: Run final repository hygiene checks**

Run:

```powershell
git diff --check
git status --short
git ls-files data cache outputs weights
rg -n -i "kuotunyu" -g '!docs/superpowers/**' .
```

Expected: clean patch, no dataset/output paths tracked, and no previous test-account reference.

- [ ] **Step 4: Repeat the browser smoke test**

Capture waiting, fall result, no-event result, error, and 390 px mobile states. Confirm no horizontal overflow, no blank result containers, no clipped rules/events, and no console errors.

- [ ] **Step 5: Review the accepted specification checklist**

Read `docs/superpowers/specs/2026-08-12-gradio-demo-uiux-design.md` section 13 and mark every item against code, tests, or a fresh screenshot. Fix any unmet item before completion.

- [ ] **Step 6: Commit any verification-only correction**

If no correction was needed, do not create an empty commit. If a verification correction changes `theme.py`, `presentation.py`, `gradio_app.py`, README, notebook, tests, script, or assets, stage the exact changed paths from that set and run:

```bash
git commit -m "Fix final demo verification findings"
```
