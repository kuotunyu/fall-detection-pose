# Gradio demo UI/UX redesign

Date: 2026-08-12

Status: Approved design; implementation pending

Primary locale: Traditional Chinese (`zh-TW`)

## 1. Objective

Redesign the local Gradio demo and its README presentation so a GitHub visitor can understand the project within seconds and perceive it as a deliberate, reproducible computer-vision system rather than a default Gradio experiment.

The interface must communicate three things in this order:

1. This project detects fall **events from pose tracks**, not isolated frames.
2. The decision is **traceable to temporal rules and thresholds**.
3. The implementation has **measured evaluation, benchmark, and test evidence**.

The public showcase will use an updated GIF and screenshots in the GitHub README. A hosted live demo and Colab-first onboarding are out of scope for this redesign. The local Gradio demo remains fully functional for recording and local evaluation.

## 2. Audience and success criteria

### Primary audience

- GitHub visitors, recruiters, and reviewers scanning the repository as a portfolio project.
- Technical visitors who want to inspect how the event decision was produced.

### Secondary audience

- A local user uploading a short video to inspect the annotated output and event JSON.

### Success criteria

- The first viewport names the method, shows real measured results, and exposes the main analysis workspace.
- A completed fall example presents the annotated video, event state, timing, rule output, and JSON download without horizontal scrolling.
- Body text is comfortable at normal browser zoom; no essential content is below 16 px on mobile.
- Empty, processing, no-event, and error states contain only information relevant to the current state. Empty result frames and blank data tables are not rendered.
- The interface remains readable at 390 px and does not turn the desktop event table into a clipped horizontal table.
- README media shows the redesigned interface and stays within repository size and dataset-license constraints.

## 3. Selected visual direction

The approved direction is a restrained scientific dashboard derived from visual iteration V10.

### Tone

- Rational, scientific, and descriptive.
- No news-style headline, marketing slogan, inflated claim, or anthropomorphic wording.
- Traditional Chinese is the default. Technical terms remain in their conventional original form where translation would reduce precision: `YOLO26-pose`, `ByteTrack`, `Track ID`, `ALARM`, `F1`, `FPS`, and `Rules fired`.

### Palette

| Role | Color | Use |
|---|---|---|
| Page background | `#DEDFD9` | Browser canvas around the app |
| App background | `#EEEBE4` | Main warm-gray surface |
| Primary surface | `#F9F8F4` | Analysis and data regions |
| Primary text | `#24312D` | Headings and values |
| Secondary text | `#5C6964` | Explanations and metadata |
| Sage | `#687F74` | Primary action, progress, successful processing |
| Sage tint | `#E7EBE5` | Rule-output background |
| Coral | `#C66C5D` | Fall/ALARM only |
| Coral tint | `#F0DED9` | Event-result background |
| Divider | `rgba(36, 49, 45, 0.16)` | Structure and data-grid separation |

Coral is semantic, not decorative. It appears only for a confirmed or high-priority fall event and error emphasis. No gradients are used.

### Typography

Use this fallback stack:

```css
"Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif
```

Use `ui-monospace, monospace` only for rule expressions, state codes, frame counts, and compact technical labels.

Desktop targets:

| Content | Size | Weight |
|---|---:|---:|
| Main project title | 34–42 px | 700 |
| Section heading | 19–21 px | 700 |
| Metric value | 29 px | 700 |
| Event state | 28 px | 700 |
| Event-strip value | 23 px | 700 |
| Rule condition | 18 px | 700 monospace |
| Body/method summary | 17–18 px | 400–500 |
| Navigation/action | 16–17 px | 700 |
| Labels/metadata | 14–15 px | 600–700 |

Mobile targets:

- Main project title: 32 px.
- Body: at least 16 px, normally 17–18 px.
- Rule conditions: at least 16 px.
- Interactive controls: at least 16 px text and 48 px touch height.

Bold is reserved for headings, readings, states, actions, and rule results. Paragraphs and descriptive labels must not be bold by default.

### Shape and depth

- Use square corners or a radius of 0–4 px for structural regions.
- Circles are limited to progress/status indicators.
- Do not use pills for tags, metadata, or navigation.
- Use dividers, alignment, and background tone before adding container outlines.
- Use one restrained shadow around the main app and a lighter shadow on the main analysis panel. Nested cards do not receive individual shadows.

## 4. Information architecture

### 4.1 Top bar

- Product label: `Fall Detection / Pose`.
- Compact `FD` mark; no invented consumer brand is required.
- Optional navigation: `方法`, `評估`, `限制`, `GitHub ↗`.
- Navigation items link to the matching README anchors in the final repository.
- Until the final repository URL is supplied, external navigation is omitted rather than linked to a placeholder.
- No previous test-account handle may be hard-coded or shown. The final repository belongs to a different GitHub account whose URL will be supplied later.

### 4.2 Method summary

Heading:

> 姿態追蹤式跌倒事件偵測

Description:

> 使用 YOLO26-pose 擷取人體姿態，ByteTrack 建立時序軌跡，再由規則式狀態機輸出可追溯的事件判定。

Technical labels:

- `YOLO26-pose`
- `ByteTrack`
- `Finite-state machine`
- `Event-level evaluation`

The heading and method summary share one row on wide screens. This prevents a large headline from leaving an unused empty block beside it.

### 4.3 Evidence strip

The evidence strip is a three-column data row, not three floating cards:

- `64.64 FPS` — `端到端速度 · T4 FP16`, model `yolo26n-pose`.
- `0.600` — `Test event-level F1`.
- `86` — `離線單元測試`.

Values are loaded from tracked project evidence (`bench.json` and `eval/metrics.json`) rather than duplicated literals. If those files are unavailable in an installed environment, omit the evidence strip; do not silently display stale fallback values.

### 4.4 Analysis workspace

Desktop layout:

- Left: annotated video, about two thirds of the workspace.
- Right: event state, timing, rule output, and processing path.
- Bottom: upload/re-run control followed by the event summary row.

The video keeps its original aspect ratio. Mockup cropping was used only to focus the visual-design review; production output must not crop video frames or obscure annotations.

The right column uses its full available height:

1. Event state (`狀態：ALARM`) and `EVENT n / total`.
2. Start time and duration.
3. Rule-output block that grows to occupy remaining height.
4. Compact pipeline: `POSE → TRACK → RULES → EVENT`.

Each rule occupies one full row. The condition is 18 px monospace and the result is placed in a fixed-width result column. Conditions must not be squeezed into several narrow horizontal cards.

### 4.5 Event summary

Replace the generic `gr.Dataframe` result with a responsive semantic event list.

For each event show:

- Track IDs.
- Start time.
- End time.
- Duration.
- `Rules fired`.

Desktop uses a single aligned data row with 23 px values. Multiple events stack vertically. Mobile uses a definition-list layout instead of horizontal scrolling. `events.json` remains available as a clear download action.

The detail view must reflect real event data. For example, `lying_persisted`, `track_lost_while_fallen`, and other rule codes are rendered from `rules_fired`; the UI must not imply that all conditions passed when an event was finalized through a different path.

## 5. Interface states

Only one primary state is visible at a time.

### 5.1 Waiting for input

- Show the method/evidence header and one upload region.
- Copy: `拖放一個短片至此處` and `或點擊選擇檔案`.
- Show supported formats, recommended duration, and size limit.
- Provide two compact examples when files are available:
  - Fall example — expected event.
  - ADL example — expected no event.
- Do not render blank output video, event table, or file-download frames.
- Model selection moves to an advanced control. `yolo26n-pose.pt` remains the default; the first-time visitor is not required to choose a model.

### 5.2 Processing

- Show determinate overall progress whenever frame count is known.
- Show the active pipeline stage and an exact frame count for pose extraction.
- Stages: video decode, pose extraction, event detection, video annotation.
- Disable duplicate submission while the request is active.
- Keep the selected filename visible.
- Estimated time is optional and is displayed only when it is based on measured progress; never fabricate an estimate.

### 5.3 Completed with fall events

- Show annotated video, `ALARM` event state, timing, real rule output, pipeline, event list, and JSON download.
- Coral indicates the confirmed fall state; the rest of the interface remains neutral.
- Re-run control copy: `重新執行`.
- Replacement input copy: `拖放另一個短片，或選擇內建範例`.

### 5.4 Completed without fall events

- Show the annotated video and an explicit conclusion: `未偵測到跌倒事件`.
- Explain: `完整影片未產生符合確認條件的 ALARM 狀態。`
- Show analyzed frames, number of tracked people, and zero event count.
- Do not render an empty event table.
- Keep the JSON download for reproducibility; its `events` array is empty.

### 5.5 Error

- Use a concise Traditional Chinese title and actionable recovery.
- Example for decode failure: `無法讀取這個影片`.
- Show a stable error code such as `VIDEO_DECODE_ERROR`, not a Python exception as the headline.
- Provide `選擇其他影片` and, where useful, `查看支援格式`.
- Put technical diagnostic details in a collapsed `診斷資訊` disclosure.
- Never display a traceback, local absolute path, token, or secret to the user.
- Temporary partial outputs are cleaned up or replaced on the next run.

## 6. Responsive behavior

Breakpoint targets are behavioral rather than tied to one device:

- Wide desktop (`≥ 1000 px`): two-column analysis workspace.
- Narrow desktop/tablet (`650–999 px`): video followed by result panel.
- Mobile (`< 650 px`): single reading column.

Mobile order:

1. Method title and compact technical context.
2. Input or annotated video.
3. Event conclusion.
4. Rule output.
5. Event details and JSON download.
6. Secondary metrics.

The desktop event grid must not be scaled down or clipped on mobile. Controls become full width. No page-level horizontal scroll is allowed at 390 px.

## 7. Component and code boundaries

Keep the inference pipeline unchanged and isolate presentation concerns.

Recommended responsibilities:

- `process_video`: existing orchestration for extract → rules → annotate → JSON. It remains independent of Gradio presentation.
- Pure formatting helpers: convert events, metrics, and runtime state into escaped presentation data.
- UI copy/constants: Traditional Chinese labels and rule-code descriptions in one local mapping.
- Theme/CSS: one dedicated stylesheet string or file for palette, typography, layout, responsive behavior, and Gradio overrides.
- `build_demo`: compose Gradio components, visibility states, advanced model control, and event wiring.
- Progress adapter: convert the existing fraction/description callbacks into the visible processing state without changing inference semantics.

Avoid one monolithic HTML string containing data formatting, application state, and CSS. Helpers must remain importable without requiring `torch`, `ultralytics`, OpenCV, or Gradio at module import time, matching the current lightweight-test architecture.

All dynamic HTML values—filenames, exception summaries, rule strings, and event fields—must be escaped before rendering.

## 8. Data flow

```text
Video + default/advanced model
        ↓
Validate input and show processing state
        ↓
process_video(..., on_progress=...)
        ↓
Annotated MP4 + FallEvent list + events.json
        ↓
Event view model
        ├─ fall events → ALARM result + event rows
        ├─ no events   → explicit zero-event result
        └─ error       → safe error state + recovery action
```

The UI is a view over existing outputs. It must not introduce a second fall-detection rule path.

## 9. README showcase

The README is the public demo surface.

### Required media

1. A short redesigned GIF showing upload/example selection, processing feedback, and a confirmed fall result.
2. A static screenshot of the ADL/no-event result.
3. Optional mobile screenshot if it remains legible at GitHub content width.

### Presentation order

- One-sentence project summary.
- Fall GIF with a precise caption.
- ADL screenshot with a precise caption.
- Compact evidence table.
- Architecture and reproduction details.

The media must use the actual pipeline output, not simulated annotations. Preserve the existing URFD attribution and license notice. Optimize each committed GIF to stay under 5 MB where practical.

Colab notebooks remain available for reproduction but are no longer presented as the primary way to experience the demo. The README does not promise a hosted live demo.

## 10. Accessibility and content rules

- Text contrast targets WCAG AA for normal text.
- Focus visibility must remain obvious after custom styling.
- Fall/error state uses icon/text plus color; coral alone is insufficient.
- Controls have visible labels and at least 48 px touch height on mobile.
- Video and screenshots receive meaningful Traditional Chinese alt text in the README.
- Do not autoplay audio.
- Avoid unnecessary animation; progress motion communicates active work only.
- Punctuation and terminology follow Traditional Chinese usage. Keep units next to values (`2.93 s`, `64.64 FPS`).

## 11. Verification strategy

### Automated tests

- Preserve existing `_events_to_rows` tests or replace them with equivalent pure-helper coverage if the Dataframe is retired.
- Test empty, single-event, multiple-event, and multiple-track event rendering.
- Test that dynamic HTML is escaped.
- Test rule-code mapping and unknown-rule fallback.
- Test evidence loading from `eval/metrics.json` and `bench.json`, including missing-file behavior.
- Test that `build_demo` can still be constructed with and without example videos.
- Keep core imports lightweight and confirm no inference dependencies are imported by formatting-helper tests.

### Browser verification

Run the local demo and capture at least:

- Desktop: 1440 × 1000 and 1800 × 1000.
- Mobile: 390 × 844.
- States: waiting, processing, fall result, no-event result, and representative error.

Verify:

- No horizontal overflow.
- No clipped event or rule text.
- No blank output containers in waiting/no-event states.
- Text meets the typography targets.
- Keyboard focus and primary actions remain usable.
- Browser console contains no UI errors.

### Repository verification

- Run the full test suite and Ruff.
- Regenerate README media from the final UI.
- Check GIF file sizes and render the README locally or through a GitHub-compatible preview.
- Search source, UI copy, README, notebooks, and configuration for the previous test-account handle before release; the result must be empty.

## 12. Scope boundaries

In scope:

- Local Gradio visual and interaction redesign.
- Responsive event presentation.
- Safe, useful loading and error states.
- Updated README GIF/screenshots and surrounding demo copy.
- Tests needed to protect the new presentation helpers and states.

Out of scope:

- Hosted deployment or authentication.
- Colab-first demo flow.
- Changes to pose extraction, tracking, fall rules, evaluation protocol, or benchmark methodology.
- New model training.
- Inventing better metrics or hiding known limitations.

## 13. Acceptance checklist

- [ ] Traditional Chinese is the primary interface language; technical terms remain precise.
- [ ] V10 palette, typography hierarchy, low-radius geometry, and scientific tone are implemented.
- [ ] Waiting state has no empty result panels.
- [ ] Processing state exposes meaningful stage/frame progress.
- [ ] Fall result uses real event fields and real `rules_fired` paths.
- [ ] No-event state gives an explicit conclusion instead of an empty table.
- [ ] Error state is actionable and does not leak internal details.
- [ ] Desktop result uses available space; rule conditions are at least 18 px and event readings 23 px.
- [ ] Mobile works at 390 px without horizontal scrolling or clipped data.
- [ ] README uses newly captured, actual-pipeline media and retains URFD attribution.
- [ ] No previous test-account handle is introduced into source, UI, README, notebooks, or configuration.
- [ ] Tests and Ruff pass.
