# fall-detection-pose | Interpretable Pose-Based Fall Event Detection

[正體中文](README.md) · **English**

[![CI](https://github.com/kuotunyu/fall-detection-pose/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/fall-detection-pose/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/kuotunyu/fall-detection-pose?color=6F877D)](https://github.com/kuotunyu/fall-detection-pose/releases/latest)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![YOLO26-pose](https://img.shields.io/badge/Pose-YOLO26--pose-635BFF)
![Tracker](https://img.shields.io/badge/Tracker-ByteTrack-287D72)
[![Code license](https://img.shields.io/badge/Code-MIT-3A7D44.svg)](LICENSE)

This project extracts human pose keypoints with **YOLO26-pose**, maintains identities with **ByteTrack**, and detects fall events through an interpretable finite-state machine (UPRIGHT → FALLING → FALLEN → ALARM). Every alert preserves its time interval, Track ID, and `rules_fired`, so the output explains not only *whether* a fall was detected, but *why* it fired.

## Evaluation summary

| Test event-level F1 | ADL specificity | T4 · yolo26n FP16 | 2-vCPU · yolo26n | Offline tests |
| ---: | ---: | ---: | ---: | ---: |
| **0.600** | **0.741** | **64.64 FPS** | **8.23 FPS** | **121** |

The test split contains 20 fall videos and 27 activities-of-daily-living (ADL) videos. Raw metrics, frozen splits, failure cases, and benchmark environment records are committed to the repository. See [Evaluation](#evaluation) for the full protocol and post-development disclosure.

## Demo

The interface keeps the annotated video, event interval, Track ID, fired rules, and processing stages in one workspace. Negative samples show an explicit **0-event** result instead of an empty table.

**`fall-06`: actual pipeline output; Track 1 reaches ALARM**

![fall-06 fall event analysis](assets/demo_fall.gif)

<details>
<summary><strong>View the ADL negative sample and narrow-screen layout</strong></summary>

**`adl-01`: 150 frames analyzed, 0 events produced**

![adl-01 no-fall result](assets/demo_adl.png)

**Narrow-screen layout**

![Narrow-screen demo](assets/demo_mobile.png)

</details>

These assets were captured from actual runs of the full Gradio pipeline. The source videos come from the [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html). See [Third-party notices](THIRD_PARTY_NOTICES.md) for the applicable terms.

## Engineering design

- **Traceable decisions:** thresholds live in [`config.yaml`](config.yaml), while every event records its `rules_fired`, time interval, and Track ID.
- **Decoupled inference and rules:** GPU pose inference runs once and writes a Keypoint Cache; tuning, state transitions, and event evaluation can then be rerun on CPU.
- **Event-level evaluation:** the project uses explicit one-to-one event matching instead of reporting frame-level accuracy alone; any event predicted on an ADL video counts as a false positive.
- **Failure-mode analysis:** beyond Precision, Recall, and F1, the repository examines tracker loss, voluntary lying down, and duplicate-event cases.
- **Reproducible engineering:** `uv.lock` pins dependencies, and GitHub Actions runs Ruff, package builds, dependency audits, and 121 offline tests on Python 3.10 and 3.12, plus one end-to-end inference smoke test with the real YOLO26n-pose and ByteTrack stack.

## Quick start

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/). On the first inference run, Ultralytics downloads the `yolo26n-pose.pt` weights, so network access is required once; later runs reuse the local cache.

### Launch the local demo

```bash
git clone https://github.com/kuotunyu/fall-detection-pose.git
cd fall-detection-pose
uv sync --locked --extra infer --extra demo
uv run python -m fall_detection.app.gradio_app --no-share
```

Open <http://127.0.0.1:7860>, upload a short video, and receive an annotated video, event summary, and `events.json`.

### Run the full pipeline

```bash
uv run fdp pipeline \
  --source input.mp4 \
  --out-dir outputs \
  --config config.yaml \
  --debug
```

Primary outputs:

| File | Contents |
| --- | --- |
| `outputs/input.parquet` | Bounding box and 17 pose keypoints for each frame/track pair |
| `outputs/input.parquet.meta.json` | Model, video hash, FPS, device, and version provenance |
| `outputs/input.events.json` | Event timing, Track ID, peak values, and `rules_fired` |
| `outputs/input_annotated.mp4` | H.264 video with skeleton, Track ID, state, and alert overlays |
| `outputs/input.debug.jsonl` | Per-frame features and state when `--debug` is enabled |

### Reproduce evaluation from the Keypoint Cache

Rerun the frozen rules and event-level evaluation on CPU without repeating pose inference:

```bash
uv run fdp evaluate \
  --cache-root /path/to/cache \
  --annotations /path/to/urfall-cam0-falls.csv \
  --model yolo26n-pose \
  --model yolo26s-pose \
  --out outputs/evaluation.json
```

`--cache-root` contains one directory per model, such as
`yolo26n-pose/fall-01.parquet`. The command reads `eval/splits.yaml` and
`config.yaml` by default and evaluates only the frozen test split; Notebook 03
retains the complete tuning history.

<details>
<summary><strong>Run development checks</strong></summary>

```bash
uv sync --locked --group dev --extra demo
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

</details>

## Architecture

This project separates one-time GPU inference from the replayable CPU decision pipeline. Model inference, rule tuning, event evaluation, and visualization can therefore be verified independently without rerunning the model after every threshold change.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '17px', 'fontFamily': 'Arial, sans-serif', 'lineColor': '#66756F'}}}%%
flowchart TB
    subgraph Interface[Input and interfaces]
        direction LR
        Video[Input video] --> Entry[Gradio Demo<br/>or fdp CLI]
        Config[(config.yaml<br/>traceable thresholds)] --> Entry
    end

    subgraph Core[Reproducible analysis core]
        direction LR
        subgraph GPU[GPU inference · run once]
            direction TB
            Pose[YOLO26-pose<br/>17 human keypoints] --> Track[ByteTrack<br/>Track ID association]
            Track --> Cache[(Keypoint Cache<br/>Parquet + provenance)]
        end

        subgraph CPU[CPU decision pipeline · replayable]
            direction TB
            Feature[Smoothed, normalized<br/>temporal / geometric features] --> FSM[Independent finite-state<br/>machine per track]
            FSM --> Post[Event merging<br/>and duration filtering]
            Post --> Events[(FallEvent<br/>events.json)]
        end

        Cache --> Feature
    end

    Entry --> Pose
    Config --> Feature

    subgraph Evidence[Outputs and validation]
        direction LR
        Render[Annotated H.264 MP4<br/>Skeleton · Track ID · state]
        Eval[Frozen test split · event-level evaluation<br/>Precision · Recall · F1 · Failure Analysis]
    end

    Cache --> Render
    Events --> Render
    Video --> Render
    Cache --> Eval
    Events --> Eval

    classDef input fill:#EBE5D9,stroke:#7C715F,stroke-width:1.5px,color:#25342F
    classDef inference fill:#DCE6EC,stroke:#5E7480,stroke-width:1.5px,color:#21343C
    classDef cache fill:#EFE2CC,stroke:#8C7452,stroke-width:1.5px,color:#382F23
    classDef decision fill:#DDE8E2,stroke:#587066,stroke-width:1.5px,color:#20352D
    classDef output fill:#E7DFE8,stroke:#77677A,stroke-width:1.5px,color:#342B36
    classDef evidence fill:#F0DED9,stroke:#91645B,stroke-width:1.5px,color:#3E2B27

    class Video,Entry,Config input
    class Pose,Track inference
    class Cache cache
    class Feature,FSM,Post decision
    class Events,Render output
    class Eval evidence
```

The Keypoint Cache is the stable boundary between GPU inference and the CPU rule engine. Its Parquet schema is versioned, while provenance metadata is both embedded in the file and written to a sidecar. Incompatible caches fail fast rather than silently contaminating evaluation results.

### Single-video analysis sequence

This sequence matches the actual call boundaries used by the Demo and `fdp pipeline`. The model produces reusable pose data; the downstream rule engine independently decides whether it constitutes a fall event.

```mermaid
%%{init: {'theme': 'base', 'sequence': {'diagramMarginX': 12, 'actorMargin': 28, 'width': 150, 'messageMargin': 32, 'noteMargin': 8, 'mirrorActors': false, 'actorFontSize': 18, 'messageFontSize': 17, 'noteFontSize': 17}, 'themeVariables': {'fontFamily': 'Arial, sans-serif', 'primaryColor': '#DDE8E2', 'primaryTextColor': '#20352D', 'primaryBorderColor': '#587066', 'lineColor': '#66756F', 'actorBkg': '#EBE5D9', 'actorBorder': '#7C715F', 'actorTextColor': '#25342F', 'signalColor': '#43564F', 'signalTextColor': '#25342F', 'noteBkgColor': '#EFE2CC', 'noteBorderColor': '#8C7452', 'noteTextColor': '#382F23'}}}%%
sequenceDiagram
    autonumber
    participant Pipeline as Demo / CLI
    participant GPU as Pose + Tracking
    participant CPU as CPU Rule Engine

    Pipeline->>GPU: extract_video(video, config)
    loop Each frame
        GPU->>GPU: YOLO26-pose + ByteTrack
    end
    GPU-->>Pipeline: Keypoint Cache<br/>Parquet + provenance metadata
    Pipeline->>CPU: run_engine(cache, config)
    loop Each Track ID
        CPU->>CPU: Feature smoothing → FSM tick → finalize
    end
    CPU-->>Pipeline: FallEvent[] + debug records
    Pipeline->>Pipeline: annotate_video(...)
    Pipeline->>Pipeline: write_events_json(...)
    Note over Pipeline,CPU: Output: H.264 MP4 · event summary · events.json
```

### Track-level state machine

Each Track ID progresses through `UPRIGHT → FALLING → FALLEN → ALARM`; `FallEvent` is the auditable event record emitted after the state machine finalizes its decision.

```mermaid
%%{init: {'theme': 'base', 'htmlLabels': false, 'flowchart': {'nodeSpacing': 55, 'rankSpacing': 70, 'curve': 'basis', 'padding': 18}, 'themeVariables': {'fontSize': '17px', 'fontFamily': 'Arial, sans-serif', 'lineColor': '#66756F', 'primaryTextColor': '#25342F', 'edgeLabelBackground': '#FAF9F6'}}}%%
flowchart TB
    Start(( ))
    UPRIGHT["UPRIGHT"]
    FALLING["FALLING"]
    FALLEN["FALLEN"]
    ALARM["ALARM"]
    FallEvent["FallEvent"]
    End(( ))
    Evidence["Interval · Track IDs<br/>Peaks · fired rules"]

    Start --> UPRIGHT
    UPRIGHT -->|"v_norm > 0.8<br/>or omega > 90°/s"| FALLING
    FALLING -->|"lying-posture vote >= 80%"| FALLEN
    FALLING -->|"unconfirmed<br/>at t = 1.2 s<br/>no event"| UPRIGHT
    FALLEN -->|"lying persists >= 0.3 s"| ALARM
    FALLEN -->|"recovery before alarm<br/>no event emitted"| UPRIGHT
    ALARM -->|"upright persists >= 0.5 s<br/>emit FallEvent"| UPRIGHT

    FALLING -->|"track ends<br/>with a final lying posture"| FallEvent
    FALLEN -->|"track ends"| FallEvent
    ALARM -->|"track / video ends"| FallEvent
    FallEvent --> End
    FallEvent -.-> Evidence

    classDef upright fill:#DDE8E2,stroke:#587066,stroke-width:1.5px,color:#20352D
    classDef falling fill:#EFE2CC,stroke:#8C7452,stroke-width:1.5px,color:#382F23
    classDef fallen fill:#E7DFE8,stroke:#77677A,stroke-width:1.5px,color:#342B36
    classDef alarm fill:#F0D8D2,stroke:#A25D50,stroke-width:2px,color:#472923
    classDef event fill:#DCE6EC,stroke:#5E7480,stroke-width:1.5px,color:#21343C
    classDef endpoint fill:#587066,stroke:#587066,color:#587066
    classDef evidence fill:#EFE2CC,stroke:#8C7452,stroke-width:1.5px,color:#382F23

    class UPRIGHT upright
    class FALLING falling
    class FALLEN fallen
    class ALARM alarm
    class FallEvent event
    class Start,End endpoint
    class Evidence evidence
```

Each Track ID owns an independent state, so one person cannot trigger another person's event. When a video ends or a track disappears, finalization uses the last state and posture to decide whether an event should be emitted, covering clips that end immediately after a fall.

## Interpretable decision logic

| Feature | Definition and purpose |
| --- | --- |
| `theta_deg` | Torso angle from vertical, measured from shoulder midpoint to hip midpoint; approximately 0° upright and 90° horizontal |
| `bbox_aspect` | Person bounding-box width divided by height |
| `h_hip` | Vertical ankle-midpoint–to–hip-midpoint gap divided by the rolling median torso length; a relative hip–ankle height cue, not a direct ground-height measurement |
| `v_norm` | Hip vertical velocity divided by the rolling median torso length and computed from actual timestamps |
| `omega` | Torso angular velocity |

False-alarm controls include:

- **Posture voting:** `theta_deg`, `bbox_aspect`, and `h_hip` use a two-of-three vote, with a required pass ratio over a rolling frame window.
- **Persistence:** FALLEN must maintain a lying posture before becoming ALARM.
- **Hysteresis:** recovery uses separate thresholds and duration requirements to avoid rapid state oscillation near a boundary.
- **Missing-data handling:** when ankles are not visible, `h_hip` does not vote; unavailable evidence is never fabricated.
- **Finalization:** the last posture is checked when a video or track ends, reducing missed events near the end of a clip.

Scale and time normalization reduce sensitivity to subject size and FPS, but they do not make the system invariant to camera viewpoint.

## Evaluation

### Protocol

- A fixed random seed of 42 defines a tune split (10 falls + 13 ADL) and a test split (20 falls + 27 ADL).
- Each ground-truth interval is expanded by **0.5 seconds** on both sides; a prediction becomes a candidate only if it has positive-duration overlap with the expanded interval.
- In ground-truth time order, the unmatched prediction with the largest overlap is selected, producing greedy one-to-one matching.
- ADL videos have no ground-truth falls; every predicted event on an ADL video is a false positive.
- All rule thresholds are selected on the tune split only; the same frozen thresholds are used to evaluate both pose models on the test split.

Frozen splits and raw results are stored in [`eval/splits.yaml`](eval/splits.yaml) and [`eval/metrics.json`](eval/metrics.json).

| Method | TP / FP / FN | Test precision | Test recall | Test F1 | ADL specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| **YOLO26n-pose (default)** | **12 / 8 / 8** | **0.600** | **0.600** | **0.600** | **0.741** |
| YOLO26s-pose | 11 / 7 / 9 | 0.611 | 0.550 | 0.579 | 0.778 |

> [!NOTE]
> After the first inspection of test results, a structural event-finalization bug was corrected and the test evaluation was run a second time. The final figures are therefore transparently reported **post-development estimates**, not a pristine one-shot holdout. Both rounds remain recorded in `eval/metrics.json`.

> [!CAUTION]
> Fall-detection papers often use different splits, temporal tolerances, and evaluation units. Results should not be placed on a shared leaderboard unless methods are rerun under the same protocol. For methodological context, see [PIFR (2025)](https://doi.org/10.1371/journal.pone.0325253) and [Núñez-Marcos et al. (2017)](https://doi.org/10.1155/2017/9474806).

## Performance benchmark

The benchmark uses a reconstructed 640×480, 30 FPS URFD `adl-01` video. The clip contains 150 frames; each configuration is run three times and the median is reported. Environment details and raw records are in [`bench.json`](bench.json).

| Environment | Model / precision | End-to-end FPS | p50 latency | p95 latency |
| --- | --- | ---: | ---: | ---: |
| NVIDIA T4 | yolo26n-pose / FP32 | 59.65 | 13.84 ms | 23.52 ms |
| NVIDIA T4 | yolo26n-pose / FP16 | **64.64** | 15.63 ms | 24.25 ms |
| 2-vCPU | yolo26n-pose / FP32 | **8.23** | 116.96 ms | 178.96 ms |
| NVIDIA T4 | yolo26s-pose / FP32 | 72.25 | 13.88 ms | 20.40 ms |
| NVIDIA T4 | yolo26s-pose / FP16 | 66.18 | 14.69 ms | 24.09 ms |
| 2-vCPU | yolo26s-pose / FP32 | 3.36 | 271.15 ms | 408.98 ms |

In this short benchmark, the smaller `n` model is not faster in every setting, and the `s` model is slower in FP16 than in FP32. Warm-up effects, shared T4 load, and measurement variance may dominate these differences, so the table should not be interpreted as a universal model-speed ranking.

## Failure analysis

Representative cases identified by reviewing the FP/FN lists in `eval/metrics.json`, feature timelines, and source frames:

- **`fall-21` (FN):** the tracker loses the subject before the fall posture fully develops; the torso angle is still low when the track ends, making track persistence the primary limitation.
- **`adl-34` (FP):** the subject voluntarily lies down and sits back up. The motion forms a sustained lying posture geometrically, but is still an FP under the ADL zero-ground-truth protocol. Pose alone cannot reliably separate intentional lying down from an accidental fall.
- **`fall-08` (duplicate-prediction FP):** the Track ID breaks during a real fall. The first segment matches the ground-truth event, while the second segment triggers again and remains unmatched, exposing a boundary between track stitching and event merging.

These cases define the system's current operating boundary and identify track continuity and event merging as priorities for further work.

## Known limitations

- This is a research and engineering prototype—not a medical device or a replacement for an emergency alert system.
- Thresholds in `config.yaml` were selected on the URFD tune split and require recalibration for new viewpoints, environments, or populations.
- Slow descents, occlusion, voluntary lying down, and complex multi-person interactions can still produce misses or false alarms.
- Track ID stability under severe multi-person occlusion and cross-dataset generalization have not been evaluated systematically.
- ONNX/TensorRT export and edge-device performance are outside the current validation scope.

<details>
<summary><strong>Reproduction workflow and notebooks</strong></summary>

The notebooks reproduce development, calibration, and evaluation; they are not required to view the demo.

| Notebook | Purpose |
| --- | --- |
| [`01_smoke_test.ipynb`](notebooks/01_smoke_test.ipynb) | End-to-end smoke test on two short videos |
| [`02_extract_urfd.ipynb`](notebooks/02_extract_urfd.ipynb) | Download URFD and build the Keypoint Cache |
| [`03_tune_eval.ipynb`](notebooks/03_tune_eval.ipynb) | Tune-split threshold search and test evaluation |
| [`04_benchmark.ipynb`](notebooks/04_benchmark.ipynb) | GPU/CPU benchmark |
| [`05_gradio_demo.ipynb`](notebooks/05_gradio_demo.ipynb) | Launch the Gradio demo in Colab |

</details>

## License and dataset

Original code written for this project is released under the [MIT License](LICENSE).
The Ultralytics package and official YOLO weights are **AGPL-3.0** by default and
are outside this project's MIT grant; review the applicable terms before a
commercial or closed-source integration.

Evaluation uses the [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html). This repository does not redistribute the original dataset. The dataset and derived demonstration media remain subject to **CC BY-NC-SA 4.0**; see [Third-party notices](THIRD_PARTY_NOTICES.md) for attribution and all third-party license boundaries.

Dataset paper: [Kwolek & Kepski, 2014](https://doi.org/10.1016/j.cmpb.2014.09.005).
