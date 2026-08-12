# FallSense：可解釋的姿態跌倒事件偵測

**正體中文** · [English](README.en.md)

[![CI](https://github.com/kuotunyu/fall-detection-pose/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/fall-detection-pose/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/kuotunyu/fall-detection-pose?color=6F877D)](https://github.com/kuotunyu/fall-detection-pose/releases/latest)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![YOLO26-pose](https://img.shields.io/badge/Pose-YOLO26--pose-635BFF)
![Tracker](https://img.shields.io/badge/Tracker-ByteTrack-287D72)
[![Code license](https://img.shields.io/badge/Code-MIT-3A7D44.svg)](LICENSE)

FallSense 以 **YOLO26-pose** 擷取人體姿態、**ByteTrack** 延續 Track ID，再由可解釋的有限狀態機（UPRIGHT → FALLING → FALLEN → ALARM）判定跌倒事件。每次警示都保留事件時間、Track ID 與 `rules_fired`，讓結果不只回答「有沒有跌倒」，也能說明「為什麼觸發」。

## 驗證摘要

| Test event-level F1 | ADL specificity | T4 · yolo26n FP16 | 2-vCPU · yolo26n | Offline tests |
| ---: | ---: | ---: | ---: | ---: |
| **0.600** | **0.741** | **64.64 FPS** | **8.23 FPS** | **121** |

測試集包含 20 段跌倒與 27 段 ADL；數據、切分名單、失敗案例與環境紀錄均已納入 repository。完整的評估限制與開發後修正揭露請見[評估結果](#評估結果)。

## Demo

介面在同一個工作區呈現標註影片、事件區間、Track ID、實際觸發規則與處理流程；沒有事件時會明確顯示 **0 個事件**，而不是留下空白結果。

**`fall-06`：實際 pipeline 輸出，Track 1 形成 ALARM 事件**

![fall-06 跌倒事件分析](assets/demo_fall.gif)

<details>
<summary><strong>查看 ADL 負例與窄螢幕版面</strong></summary>

**`adl-01`：分析 150 frames，維持 0 個事件**

![adl-01 無跌倒事件](assets/demo_adl.png)

**窄螢幕版面**

![窄螢幕 Demo](assets/demo_mobile.png)

</details>

展示媒體皆由 Gradio 介面實際執行完整 pipeline 後擷取。影片來源為 [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)，適用條款見 [Third-party notices](THIRD_PARTY_NOTICES.md)。

## 工程設計重點

- **可追溯的決策**：閾值集中於 [`config.yaml`](config.yaml)，事件輸出同時記錄 `rules_fired`、時間與 Track ID。
- **推論與規則解耦**：GPU 姿態推論只需執行一次並寫入 Keypoint Cache；調參、狀態判定與事件評估可直接在 CPU 重跑。
- **事件層級評估**：採用明確的一對一事件配對，而不是只報告 frame-level accuracy；ADL 中的任何預測都會計為 FP。
- **失敗模式分析**：除 Precision、Recall 與 F1 外，也分析追蹤中斷、主動臥床及重複事件等具體案例。
- **可重現工程流程**：`uv.lock` 鎖定依賴，GitHub Actions 在 Python 3.10／3.12 執行 Ruff、封裝驗證、依賴安全稽核與 121 項 offline tests，另以真實 YOLO26n-pose 與 ByteTrack 執行 1 項端到端 inference smoke test。

## 快速開始

需要 Python 3.10+ 與 [`uv`](https://docs.astral.sh/uv/)。首次推論時，Ultralytics 會下載 `yolo26n-pose.pt` 權重，因此需要網路連線；後續可重用本機快取。

### 啟動本機 Demo

```bash
git clone https://github.com/kuotunyu/fall-detection-pose.git
cd fall-detection-pose
uv sync --locked --extra infer --extra demo
uv run python -m fall_detection.app.gradio_app --no-share
```

開啟 <http://127.0.0.1:7860>，上傳短片即可取得標註影片、事件摘要與 `events.json`。

### 執行完整 Pipeline

```bash
uv run fdp pipeline \
  --source input.mp4 \
  --out-dir outputs \
  --config config.yaml \
  --debug
```

主要輸出：

| 檔案 | 內容 |
| --- | --- |
| `outputs/input.parquet` | 每個 frame／track 的 bounding box 與 17 個 pose keypoints |
| `outputs/input.parquet.meta.json` | 模型、影片雜湊、FPS、裝置與版本出處 |
| `outputs/input.events.json` | 事件時間、Track ID、峰值與 `rules_fired` |
| `outputs/input_annotated.mp4` | H.264 骨架、Track ID、狀態與警示標註影片 |
| `outputs/input.debug.jsonl` | `--debug` 啟用時輸出的逐幀特徵與狀態 |

### 從 Keypoint Cache 重現評估

推論產生的 cache 可直接在 CPU 上重跑凍結規則與 event-level 評估，不必再次執行模型：

```bash
uv run fdp evaluate \
  --cache-root /path/to/cache \
  --annotations /path/to/urfall-cam0-falls.csv \
  --model yolo26n-pose \
  --model yolo26s-pose \
  --out outputs/evaluation.json
```

`--cache-root` 內應依模型分目錄，例如 `yolo26n-pose/fall-01.parquet`。命令預設讀取
`eval/splits.yaml` 與 `config.yaml`，只評估凍結的 test split；Notebook 03 則保留完整調參歷程。

<details>
<summary><strong>執行開發驗證</strong></summary>

```bash
uv sync --locked --group dev --extra demo
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

</details>

## 系統架構

FallSense 將一次性的 GPU inference 與可重複執行的 CPU decision pipeline 分開；這讓模型推論、規則調整、事件評估與視覺化可以各自驗證，不必每次改閾值都重新跑模型。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '17px', 'fontFamily': 'Arial, sans-serif', 'lineColor': '#66756F'}}}%%
flowchart TB
    subgraph Interface[輸入與操作介面]
        direction LR
        Video[輸入影片] --> Entry[Gradio Demo<br/>或 fdp CLI]
        Config[(config.yaml<br/>可追溯閾值)] --> Entry
    end

    subgraph Core[可重現分析核心]
        direction LR
        subgraph GPU[GPU inference · 僅執行一次]
            direction TB
            Pose[YOLO26-pose<br/>17 個人體關鍵點] --> Track[ByteTrack<br/>Track ID 關聯]
            Track --> Cache[(Keypoint Cache<br/>Parquet + provenance)]
        end

        subgraph CPU[CPU decision pipeline · 可重複執行]
            direction TB
            Feature[平滑與正規化<br/>時序／幾何特徵] --> FSM[每個 Track 獨立<br/>有限狀態機]
            FSM --> Post[事件合併<br/>與最短時長過濾]
            Post --> Events[(FallEvent<br/>events.json)]
        end

        Cache --> Feature
    end

    Entry --> Pose
    Config --> Feature

    subgraph Evidence[輸出與驗證]
        direction LR
        Render[標註輸出 · H.264 MP4<br/>骨架 · Track ID · 狀態]
        Eval[凍結 test split · event-level 評估<br/>Precision · Recall · F1 · Failure Analysis]
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

Keypoint Cache 是 GPU inference 與 CPU rule engine 之間的穩定介面。Parquet schema 具有版本檢查，metadata 同時嵌入檔案並寫入 sidecar；不相容的 cache 會 fail fast，避免版本漂移污染評估結果。

### 單次影片分析時序

以下時序對應 Demo 與 `fdp pipeline` 的實際呼叫邊界；模型只負責產生可重用的姿態資料，是否形成事件則由後續規則引擎決定。

```mermaid
%%{init: {'theme': 'base', 'sequence': {'diagramMarginX': 12, 'actorMargin': 28, 'width': 150, 'messageMargin': 32, 'noteMargin': 8, 'mirrorActors': false, 'actorFontSize': 18, 'messageFontSize': 17, 'noteFontSize': 17}, 'themeVariables': {'fontFamily': 'Arial, sans-serif', 'primaryColor': '#DDE8E2', 'primaryTextColor': '#20352D', 'primaryBorderColor': '#587066', 'lineColor': '#66756F', 'actorBkg': '#EBE5D9', 'actorBorder': '#7C715F', 'actorTextColor': '#25342F', 'signalColor': '#43564F', 'signalTextColor': '#25342F', 'noteBkgColor': '#EFE2CC', 'noteBorderColor': '#8C7452', 'noteTextColor': '#382F23'}}}%%
sequenceDiagram
    autonumber
    participant Pipeline as Demo / CLI
    participant GPU as Pose + Tracking
    participant CPU as CPU Rule Engine

    Pipeline->>GPU: extract_video(video, config)
    loop 每個 frame
        GPU->>GPU: YOLO26-pose + ByteTrack
    end
    GPU-->>Pipeline: Keypoint Cache<br/>Parquet + provenance metadata
    Pipeline->>CPU: run_engine(cache, config)
    loop 每個 Track ID
        CPU->>CPU: 特徵平滑 → FSM tick → finalize
    end
    CPU-->>Pipeline: FallEvent[] + debug records
    Pipeline->>Pipeline: annotate_video(...)
    Pipeline->>Pipeline: write_events_json(...)
    Note over Pipeline,CPU: 輸出：H.264 MP4 · 事件摘要 · events.json
```

### Track-level 狀態機

每個 Track ID 依 `UPRIGHT → FALLING → FALLEN → ALARM` 演進；`FallEvent` 是狀態機完成判定後輸出的可稽核事件紀錄。

```mermaid
%%{init: {'theme': 'base', 'htmlLabels': false, 'themeVariables': {'fontSize': '17px', 'fontFamily': 'Arial, sans-serif', 'lineColor': '#66756F', 'primaryTextColor': '#25342F', 'edgeLabelBackground': '#FAF9F6', 'noteBkgColor': '#EFE2CC', 'noteBorderColor': '#8C7452', 'noteTextColor': '#382F23'}}}%%
stateDiagram-v2
    direction TB
    %% En spaces compensate for GitHub's state-node text metrics.
    state " UPRIGHT " as UPRIGHT
    state " FALLING " as FALLING
    state " FALLEN " as FALLEN
    state " ALARM " as ALARM
    state " FallEvent " as FallEvent
    [*] --> UPRIGHT
    UPRIGHT --> FALLING: v_norm > 0.8<br/>∨ omega > 90°/s
    FALLING --> FALLEN: 躺姿投票 >= 80%
    FALLING --> UPRIGHT: 未確認<br/>t = 1.2 s<br/>不輸出事件
    FALLEN --> ALARM: 躺姿持續 >= 0.3 s
    FALLEN --> UPRIGHT: 警示前持續回正<br/>不輸出事件
    ALARM --> UPRIGHT: 回正持續 >= 0.5 s<br/>輸出 FallEvent

    FALLING --> FallEvent: track 結束<br/>且末次姿態為躺姿
    FALLEN --> FallEvent: track 結束
    ALARM --> FallEvent: track / 影片結束
    FallEvent --> [*]

    note right of FallEvent
        區間 · Track IDs
        特徵峰值 · 觸發規則
    end note

    classDef upright fill:#DDE8E2,stroke:#587066,stroke-width:1.5px,color:#20352D
    classDef falling fill:#EFE2CC,stroke:#8C7452,stroke-width:1.5px,color:#382F23
    classDef fallen fill:#E7DFE8,stroke:#77677A,stroke-width:1.5px,color:#342B36
    classDef alarm fill:#F0D8D2,stroke:#A25D50,stroke-width:2px,color:#472923
    classDef event fill:#DCE6EC,stroke:#5E7480,stroke-width:1.5px,color:#21343C

    class UPRIGHT upright
    class FALLING falling
    class FALLEN fallen
    class ALARM alarm
    class FallEvent event
```

每個 Track ID 都有獨立狀態，不會因另一個人觸發警示而共用事件。影片結束或 track 消失時，finalization 會依最後狀態決定是否輸出事件，處理跌倒後片段立即結束的情況。

## 可解釋的判定依據

| 特徵 | 定義與用途 |
| --- | --- |
| `theta_deg` | 肩膀中點至髖部中點相對鉛直方向的軀幹角度；直立約 0°、橫躺約 90° |
| `bbox_aspect` | 人體 bounding box 的寬高比 |
| `h_hip` | 踝部中點與髖部中點的垂直差，除以滑動中位數軀幹長度；用來表示髖踝相對高度，而非直接量測離地高度 |
| `v_norm` | 髖部垂直速度除以滑動中位數軀幹長度，以實際時間差分計算 |
| `omega` | 軀幹角度變化率 |

主要防誤報設計：

- **Posture voting**：`theta_deg`、`bbox_aspect` 與 `h_hip` 採三取二，並要求滑動視窗內足夠比例的 frames 通過。
- **Persistence**：進入 FALLEN 後仍需持續躺姿，才會形成 ALARM。
- **Hysteresis**：回復直立採用不同門檻與持續時間，避免狀態在臨界值來回跳動。
- **Missing-data handling**：踝部不可見時，`h_hip` 不參與投票；系統不會捏造不可得的特徵。
- **Finalization**：影片或 track 結束時檢查末端姿態，降低片尾事件漏報。

尺度與時間正規化可降低人物大小及 FPS 差異造成的影響，但不代表對攝影機視角具有不變性。

## 評估結果

### 評估協定

- 使用固定 random seed 42，分為 tune split（10 falls + 13 ADL）與 test split（20 falls + 27 ADL）。
- Ground-truth 區間前後各擴張 **0.5 秒**；預測與擴張後區間有正時間交集才列為候選。
- 依 ground truth 的時間順序，選擇尚未使用且交集最大的預測，執行 greedy one-to-one matching。
- ADL 影片沒有 ground-truth fall；任何預測事件都計為 FP。
- 所有規則閾值僅使用 tune split 搜尋；同一組凍結閾值用於兩個 pose 模型的 test 評估。

切分名單與原始結果位於 [`eval/splits.yaml`](eval/splits.yaml) 與 [`eval/metrics.json`](eval/metrics.json)。

| 方法 | TP / FP / FN | Test precision | Test recall | Test F1 | ADL specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| **YOLO26n-pose（預設）** | **12 / 8 / 8** | **0.600** | **0.600** | **0.600** | **0.741** |
| YOLO26s-pose | 11 / 7 / 9 | 0.611 | 0.550 | 0.579 | 0.778 |

> [!NOTE]
> 第一次查看 test 結果後，專案修正了事件 finalization 的結構性錯誤，再執行第二輪評估。因此最終數字是透明揭露的 **post-development estimates**，不應解讀為完全未觸碰的一次性 holdout。完整兩輪紀錄保留於 `eval/metrics.json`。

> [!CAUTION]
> 跌倒偵測文獻常使用不同的資料切分、時間容忍範圍與評估單位；若未在相同協定下重跑，不應將數字放入同一 leaderboard 直接排名。相關方法背景可參考 [PIFR（2025）](https://doi.org/10.1371/journal.pone.0325253) 與 [Núñez-Marcos et al.（2017）](https://doi.org/10.1155/2017/9474806)。

## 效能測試

Benchmark 使用 URFD `adl-01` 重建的 640×480、30 FPS 影片。該影片實際只有 150 frames；每個設定執行 3 次並取中位數。環境與原始紀錄位於 [`bench.json`](bench.json)。

| 環境 | 模型 / 精度 | End-to-end FPS | p50 latency | p95 latency |
| --- | --- | ---: | ---: | ---: |
| NVIDIA T4 | yolo26n-pose / FP32 | 59.65 | 13.84 ms | 23.52 ms |
| NVIDIA T4 | yolo26n-pose / FP16 | **64.64** | 15.63 ms | 24.25 ms |
| 2-vCPU | yolo26n-pose / FP32 | **8.23** | 116.96 ms | 178.96 ms |
| NVIDIA T4 | yolo26s-pose / FP32 | 72.25 | 13.88 ms | 20.40 ms |
| NVIDIA T4 | yolo26s-pose / FP16 | 66.18 | 14.69 ms | 24.09 ms |
| 2-vCPU | yolo26s-pose / FP32 | 3.36 | 271.15 ms | 408.98 ms |

這組短 benchmark 中，較小的 `n` 模型沒有在每種設定都更快，`s` 模型的 FP16 也慢於 FP32；結果可能受 warm-up、共享 T4 負載與量測波動影響，不宜解讀為普遍的模型速度排序。

## 失敗分析（Failure Analysis）

依 `eval/metrics.json` 的 FP／FN 清單回看特徵時序與畫面後，代表性案例包括：

- **`fall-21`（FN）**：追蹤器在跌倒姿態完全形成前遺失目標；軀幹角度仍低時 track 已中斷，限制主要來自追蹤持續度。
- **`adl-34`（FP）**：受測者主動躺下再坐起。幾何上確實形成持續躺姿，但依 ADL 的 0-GT 協定仍計為 FP，顯示單靠姿態難以區分「主動臥床」與「意外跌倒」。
- **`fall-08`（重複預測 FP）**：真實跌倒期間 Track ID 斷裂；第一段已正確配對，第二段再次觸發成為未配對預測，揭示 track stitching 與 event merging 的邊界問題。

這些案例界定了系統目前的適用邊界，也指出後續應優先改善 track continuity 與 event merging。

## 已知限制

- 本專案是研究與工程驗證用 prototype，不是醫療器材，也不能取代緊急通報系統。
- `config.yaml` 的閾值由 URFD tune split 選出；更換攝影機角度、場域或族群時需要重新校準。
- 慢速倒地、遮擋、主動躺下及複雜多人互動仍可能造成漏報或誤報。
- 多人嚴重遮擋下的 Track ID 穩定性與跨資料集 generalization 尚未完成系統性驗證。
- ONNX／TensorRT 匯出與 edge device 效能不在目前的驗證範圍。

<details>
<summary><strong>重現流程與 Notebooks</strong></summary>

Notebooks 用於重現開發、校準與評估流程，不是觀看 Demo 的必要步驟。

| Notebook | 用途 |
| --- | --- |
| [`01_smoke_test.ipynb`](notebooks/01_smoke_test.ipynb) | 兩段短片的 end-to-end smoke test |
| [`02_extract_urfd.ipynb`](notebooks/02_extract_urfd.ipynb) | 下載 URFD 並建立 Keypoint Cache |
| [`03_tune_eval.ipynb`](notebooks/03_tune_eval.ipynb) | Tune split 閾值搜尋與 test 評估 |
| [`04_benchmark.ipynb`](notebooks/04_benchmark.ipynb) | GPU／CPU benchmark |
| [`05_gradio_demo.ipynb`](notebooks/05_gradio_demo.ipynb) | 在 Colab 啟動 Gradio Demo |

</details>

## 授權與資料來源

本專案作者撰寫的原始程式碼以 [MIT License](LICENSE) 釋出。Ultralytics 套件與官方
YOLO 權重預設適用 **AGPL-3.0**，不屬於本專案 MIT 授權範圍；商業或封閉原始碼整合前
應確認其適用授權。

評估使用 [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)。本 repository 不重新散布原始資料；資料集與衍生展示媒體依 **CC BY-NC-SA 4.0** 使用。完整第三方授權邊界見 [Third-party notices](THIRD_PARTY_NOTICES.md)。

資料集論文：[Kwolek & Kepski, 2014](https://doi.org/10.1016/j.cmpb.2014.09.005)。
