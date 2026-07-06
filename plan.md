# fall-detection-pose — v1 實作計畫

> 本檔為 v1 的工作計畫(已與專案擁有者確認核准),實作照此進行,期間可隨時討論修改。
> 計畫中的查證事實均來自 2026-07-06 的官方一手來源(連結見各節)。

## Context

求職作品集專案:**不訓練模型**,用 Ultralytics YOLO26 pose 預訓練權重 + 內建 ByteTrack,做影片跌倒偵測。賣點不是模型創新,而是工程能力:可解釋的規則引擎、event-level 誠實評估、失敗分析、可重現性。GitHub 上同類 repo 極多但通病是「零量化評估、magic number 閾值、只秀成功 demo、mp4v 影片瀏覽器播不出來」——本專案逐一反制。

## 已確認決策

| 項目 | 決定 |
|---|---|
| 文件語言 | 正體中文為主,專有名詞用原文(keypoint、track id…);commit 訊息英文 |
| 執行環境 | **全部 Colab**(「Run all」notebook);benchmark 為 **Colab GPU + Colab CPU** |
| 本機角色 | 僅一個輕量 venv(uv:numpy/pandas/pyarrow/pytest,**無 torch**)供 push 前跑規則引擎單元測試 |
| 調參協定 | tune/test 切分:**tune = 10 falls + 13 ADL;test = 20 falls + 27 ADL**(seed=42 分層,名單進版控 `eval/splits.yaml`) |
| GitHub | `fall-detection-pose`,public |
| 模型 | YOLO26-pose(已確認正式釋出);config 保留模型名參數 |
| 延伸模組(工地安全帽) | 選做,另行決定;v1 完全不碰 |

## 查證結果(來源皆官方/一手)

- **YOLO26-pose 已釋出**:`yolo26n-pose.pt` / `yolo26s-pose.pt` 在 ultralytics/assets **v8.4.0**(2026-01-13)release assets 中。COCO-pose mAP(50-95):n=57.2、s=63.0。→ pin `ultralytics>=8.4`。
- **API**:`model.track(frame, tracker='bytetrack.yaml', persist=True, conf=…, iou=…)`;讀 `results[0].keypoints.xy / .conf`(**conf 可為 None**)、`results[0].boxes.id`(**可為 None**);COCO 17 keypoints。`persist=True` 只用在自己逐幀餵的迴圈。YOLO26 是 NMS-free 端到端,conf 分布與 YOLO11 不同 → conf 閾值在 tune split 校準,不抄舊教學。
- **URFD**:官方頁 `https://fenix.ur.edu.pl/~mkepski/ds/uf.html`(舊網域 fenix.univ.rzeszow.pl 已死)。逐檔下載 `…/ds/data/fall-01-cam0-rgb.zip`,無打包檔;cam0 RGB 全部約 **4.5GB**。cam0=平行地面側視(用這個)、cam1=天花板俯視;**ADL 只有 cam0**。640×480 @30fps PNG 序列。標註 `urfall-cam0-falls.csv` / `urfall-cam0-adls.csv`:`label ∈ {-1 未躺, 0 跌落中, 1 躺地}`。授權 **CC BY-NC-SA 4.0**(資料不進 git、不重傳);引用:Kwolek & Kepski 2014, CMPB, DOI 10.1016/j.cmpb.2014.09.005。Kaggle 鏡像授權標示 Unknown → 只作備援。
- **Gradio 現行穩定版是 6.x**(6.19.0)→ pin `gradio>=6,<7`;5.x 範例語法在 6 會壞。OpenCV `mp4v` 瀏覽器不可播、pip opencv 無 H.264 encoder → **一律 ffmpeg `-c:v libx264 -pix_fmt yuv420p -movflags +faststart` 重編碼**。
- **文獻閾值參考**:軀幹傾角 45°(Chen)~60°(Ambianic);bbox 寬高比 >1.0~1.4;kpt conf 門檻 0.15~0.5;GMDCSA 規則法在 URFD:sens 91.67 / **spec 72.50**(ADL 刻意含躺下);PIFR 2025:P 88.8 / R 94.1 / F1 91.4;CNN 上限 Núñez-Marcos 2017 acc 98.63。**arXiv 2503.19501 已撤稿,不得引用**。像素閾值不可跨解析度遷移 → 一律用軀幹長正規化。

## 執行模型(Colab 協作迴圈)

```
開發(本機):寫 code + notebook → pytest(輕量 venv)→ git push
執行(Colab):開 notebook → Runtime → Run all → 產物落 Google Drive
            → notebook 末格 print 關鍵 JSON 摘要,貼回對話
→ 依結果調整 → 下一輪
```

- 每本 notebook 開頭:`git clone` repo → `pip install -e .` → mount Drive;資料/快取/輸出都在 `Drive/fall-detection-pose/{data,cache,outputs}`(VM 暫時性對策)。
- 下載與抽取腳本全部 **idempotent**(檔案存在即跳過),Colab 斷線重跑不重工。
- 關鍵設計:**推論與規則引擎解耦**——GPU 只跑一次 `extract` 把 keypoints 落成 parquet cache 存 Drive,之後所有調參/評估/失敗分析都是秒級 CPU 工作(甚至免 GPU runtime)。

## Repo 結構

```
fall-detection-pose/
├── pyproject.toml            # uv;核心依賴輕量,infer/demo/plot 為 extras
├── config.yaml               # 全部閾值 + 每項一行理由註解
├── plan.md                   # 本計畫
├── .gitignore                # data/ cache/ outputs/ weights/ .venv/
├── LICENSE                   # 程式碼 MIT;URFD 資料 CC BY-NC-SA 4.0 不隨 repo 散布
├── README.md
├── src/fall_detection/
│   ├── config.py             # pydantic 載入/驗證 config.yaml(exit>enter 等非法值 fail-fast)
│   ├── cli.py                # argparse 子命令:extract / detect / annotate / eval / tune / bench
│   ├── io/
│   │   ├── video.py          # cv2 讀寫封裝;mp4v 暫存 → ffmpeg libx264 重編碼
│   │   ├── cache.py          # keypoint cache parquet + meta 讀寫、schema_version 嚴格比對
│   │   └── urfd.py           # 官方站逐檔下載(retry/skip-existing)、PNG zip→30fps mp4、CSV 解析
│   ├── inference/
│   │   ├── pose_tracker.py   # YOLO26-pose + bytetrack 封裝;guard boxes.id / keypoints.conf 為 None
│   │   └── extract.py        # 影片(批次) → cache parquet【唯一 GPU 步驟】
│   ├── rules/
│   │   ├── features.py       # 單幀幾何特徵(純函式、無狀態、可單測)
│   │   ├── smoothing.py      # 滑動中位數、hold-last with TTL
│   │   ├── state_machine.py  # 每 track FSM:遲滯 + 去抖動 + 超時
│   │   └── engine.py         # cache rows → track 時序 → FSM → 事件;含 track 縫合
│   ├── events/schema.py      # FallEvent dataclass、events.json 序列化、merge/min-duration 後處理
│   ├── viz/annotate.py       # 骨架+track id+狀態+ALARM 疊加 → H.264 mp4
│   ├── eval/
│   │   ├── ground_truth.py   # urfall-cam0-*.csv → GT 事件區間
│   │   ├── matching.py       # event-level 配對、P/R/F1/Specificity
│   │   └── report.py         # metrics.json → markdown 表;FP/FN 特徵曲線+關鍵幀匯出
│   ├── bench/benchmark.py    # model × device FPS 矩陣 → bench.json
│   └── app/gradio_app.py     # Gradio 6 demo
├── notebooks/                # 皆為 Colab「Run all」可跑完
│   ├── 01_smoke_test.ipynb   # 2 段短片端到端煙測(先小後大)
│   ├── 02_extract_urfd.ipynb # URFD 全量下載→Drive + n/s 兩模型抽 cache【GPU】
│   ├── 03_tune_eval.ipynb    # tune split 網格調參 → 凍結 → test split 定稿 + 失敗分析【CPU 可】
│   ├── 04_benchmark.ipynb    # FPS 矩陣【GPU runtime,CPU 欄同 VM 量】
│   └── 05_gradio_demo.ipynb  # share=True 臨時連結 + 錄 demo GIF
├── tests/                    # synthetic.py + test_features / test_state_machine / test_matching / test_cache
└── eval/splits.yaml          # tune/test 名單(seed=42,進版控)
```

原則:`rules/`、`events/`、`eval/` **不 import torch/ultralytics/cv2**(只吃 numpy+pandas rows)→ 本機輕量 venv 可完整測試。

## Keypoint cache schema(parquet)

每列 = (frame, track) 偵測;無偵測幀不落列。欄位:`frame_idx:int32`、`t_ms:float64`、`track_id:int32`(`boxes.id is None` → -1)、`bbox_x1..y2:float32`、`bbox_conf:float32`、`kpts_xy:list<float32>[34]`、`kpts_conf:list<float32>[17]`(None → 整列 -1.0 哨兵)。
Meta(parquet metadata + 同名 `.meta.json` 雙寫):`schema_version`、`video_path`、`video_sha1`、`fps`、`width/height`、`n_frames`、`model_name`、`ultralytics_version`、`tracker_yaml`、`conf`、`iou`、`device`、`git_commit`。`detect` 讀到不相容 schema_version 直接報錯要求重 extract,不隱式降級。

## 規則引擎

**特徵**(影像座標 y 向下;conf ≥ `kpt_conf_min` 才算可見;肩 5,6、髖 11,12、踝 15,16):
- 中點 S/H/A = 可見肩/髖/踝均值;肩或髖全不可見 → 該幀 invalid。
- 軀幹長 `L = ‖H−S‖`,取滑動中位數 `L̃` 當尺度單位 → **解析度/距離不變**。
- 軀幹傾角 `θ = atan2(|H.x−S.x|, H.y−S.y)`(直立≈0°、躺平≈90°;頭低於髖 clamp 90°)。
- bbox 寬高比 `r = w/h`;髖踝相對高度 `h_hip = (A.y−H.y)/L̃`(站立 1.5~2.5,躺地→0;踝不可見則 invalid)。
- 垂直速度 `v = ΔH.y/(L̃·Δt)` 單位「軀幹長/秒」→ **fps 不變**;H.y 先過 3 點中位數濾波。
- invalid 幀 hold-last(TTL=`max_kpt_gap_s`);θ/r/h_hip 各過 `smooth_s` 滑動中位數。

**狀態機**(每 track 一台):
```
UPRIGHT ─(v>v_fall_enter 或 dθ/dt>omega_enter)→ FALLING ─(躺姿 2-of-3 投票×窗內 80%)→ FALLEN ─(持續 t_confirm)→ ALARM
   ▲            └(t_falling_timeout 內未確認 → 回退,不出事件:擋快速坐下/蹲下)
   └──(θ<θ_exit 且 h_hip>exit 持續 t_recover;遲滯 exit<enter)──┘
```
躺姿投票 = `[θ>60°, r>1.0, h_hip<0.5]` 三取二(單一特徵各有失效視角,GMDCSA 單規則 spec 72.5% 的教訓)。事件 = 進入 FALLEN;起點 = 進 FALLING 幀。
**track 縫合**:新 track 與 `track_stitch_window_s` 內消失舊 track 的末 bbox IoU ≥ 0.3 → 繼承 FSM 狀態,事件記 `track_ids:[old,new]`(跌倒瞬間形變大,ByteTrack 常斷 id;在 engine 層做,不魔改 tracker)。

**config.yaml**:全部閾值 + 理由見 [config.yaml](config.yaml)。**現值為文獻起點,最終值由 notebook 03 在 tune split 校準後凍結,校準過程記錄成 config 註解素材。**

**events.json**:每事件 `{track_ids, start_frame, end_frame, start_time_s, end_time_s, peak_features:{max_v, max_theta}, rules_fired}`。另 `--debug` 輸出 per-frame 特徵 JSONL(失敗分析地基)。

## URFD 評估協定(寫死進 README)

- **GT 事件 = [第一個 label=0 幀, 其後連續 label=1 區段末幀]**(falling+lying;文獻對正類定義分歧,明示慣例)。每支 fall 影片恰 1 事件;ADL 影片 0 事件。
- 配對:預測與 GT(±0.5s tolerance)**任何時間交集**即候選,每 GT 貪婪配最大交集一個預測。TP=配對 GT 數;FN=未配對 GT;FP=fall 影片多餘預測 + ADL 影片全部預測(事件碎裂會被懲罰)。
- 指標:P、R、F1 + video-level Specificity(對齊文獻報法)。
- `tune` 子命令程式層面只接受 tune split;README 主表報 test,附表報 tune 與 all-70(附小樣本警語)。表格數字一律由 metrics.json 生成,不手填。
- 失敗分析 **2 FP + 1 FN**:每例 = 6-8 幀 frame strip + 特徵時序圖(θ/h_hip/v 曲線疊閾值虛線、GT 紅帶、觸發藍帶)+ 短文分析。預期案例:ADL 刻意躺下(FP)、快速坐下(FP)、躺地 keypoint dropout(FN)。

## Benchmark(Colab 版)

- 矩陣:`{yolo26n-pose, yolo26s-pose} × {Colab GPU, Colab CPU}`;GPU 另附 half=True 欄。
- 固定一支 URFD 重組 640×480 mp4 取 300 幀、**先全部解碼進記憶體**;warmup 20 幀;GPU 計時夾 `torch.cuda.synchronize()`;每格 3 次取中位數;分報**純推論 FPS** 與**端到端 FPS**、p50/p95 延遲。
- `bench.json` 記錄 GPU 型號、CPU model/threads(Colab 約 2 vCPU,數字會低——照實報)、torch/ultralytics 版本、git commit。README 註明:benchmark 腳本可攜,任何機器可補跑一列;**不引用官方「CPU 快 43%」**(那是 detect ONNX 數字,不適用 pose)。

## Gradio demo(Gradio 6)

`gr.Blocks` + 上傳 `gr.Video(sources=["upload"])` → Process 按鈕 → `gr.Video`(標註影片)+ `gr.Dataframe`(事件表)+ `gr.File`(events.json 下載)。`gr.Progress` 包幀迴圈;`launch(max_file_size="200mb", share=True)`;processing 事件 `concurrency_limit=1`。輸出影片自行 ffmpeg 轉 H.264,`format="mp4"` 僅作保險。**用 6.x 語法,不抄 5.x 範例**。

## 單元測試(合成軌跡,不碰模型/影片)

`tests/synthetic.py` 參數化骨架生成 cache 同格式 rows。案例:教科書跌倒恰 1 事件、走動 0 事件、快速坐下被 timeout 攔下、緩慢躺下 0 事件(README 承認為設計取捨)、躺地中 dropout 不裂成兩事件、超 TTL 正常截斷、track 縫合 `track_ids==[1,7]`、None/哨兵值不炸、**尺度不變**(座標×0.5 事件相同)、**fps 不變**(30↔15fps 起訖差<0.2s)、遲滯不抖動、matching 邊界案例、cache roundtrip。本機輕量 venv 跑,push 前必綠;notebook 01 也 `!pytest` 一次。

## README 大綱(正體中文)

1. 標題 + 一句話定位(強調工程:規則可解釋、event-level 誠實評估、失敗分析)+ badges → 2. **兩張 demo GIF 並排**(跌倒正確觸發 / ADL 不誤觸——展示不誤觸才是懂 precision 的訊號)→ 3. Results at a Glance 小表 → 4. mermaid 架構圖(含 config.yaml 與 debug JSONL 節點)→ 5. 判斷邏輯(特徵數學 + FSM 圖 + 閾值表連 config)→ 6. Quick Start(Colab badge 連 notebooks)→ 7. 評估(**先協定後數字**;文獻對照表分開放並註明協定不可直比:GMDCSA 62.16、PIFR 91.4、CNN 98.63)→ 8. Benchmark → 9. 失敗分析 → 10. 限制與未來工作(遮擋、多人重疊、鏡頭角度、慢速跌倒、Re-ID、跨資料集泛化、TensorRT 匯出)→ 11. 資料集出處與授權(URFD 引用 + CC BY-NC-SA 聲明、不重散布)→ 12. 結構/License。
GIF:ffmpeg palettegen 壓 480p <5MB 直接進 git。面試 10 題預判清單當 README 驗收標準(每題都要能指著某節回答)。

## 里程碑(每個 = git commit,英文訊息)

| # | 內容 | 驗證 |
|---|---|---|
| M0 | repo init:pyproject(uv)、config.yaml、.gitignore、LICENSE、plan.md、README stub;git init → GitHub public → push | 本機 `uv sync` 輕量 venv 成功 |
| M1 | rules/ + events/ + eval/matching + cache schema + 全部合成軌跡單測 | 本機 pytest 全綠 |
| M2 | inference/ + viz/ + cli + io/urfd + **notebook 01 煙測**(2 短片:1 fall + 1 ADL) | Colab Run all:標註 mp4 可播(H.264)、骨架/track id/狀態正確、events.json 合理 |
| M3 | **notebook 02**:URFD 全量下載→Drive、重組 mp4、n/s 兩模型抽 cache;GT 解析 + splits.yaml | cache 覆蓋 70 支、meta 完整;GT 事件數 = 30 |
| M4 | **notebook 03**:tune split 網格調參 → 凍結 config → test split 定稿 metrics.json + 失敗分析 artifacts | metrics 摘要回報;挑 2FP+1FN 成文 |
| M5 | **notebook 04** benchmark | bench.json 四格齊 |
| M6 | **notebook 05** Gradio demo + 錄兩張 GIF | share 連結可用、上傳→回標註影片+事件表 |
| M7 | README 定稿(數字全部由 metrics.json/bench.json 生成)、最終潤稿 | 面試 10 題逐一可答;連結全部有效 |

## 風險與緩解(摘要)

躺地 keypoint dropout(pose 模型偏訓直立人形)→ hold-last TTL + 2-of-3 投票 + merge_gap,失敗分析展示實例。URFD ADL 刻意含躺下 → 速度/角速度作必要觸發,README 引 GMDCSA 設定期望。跌倒瞬間 ByteTrack 斷 id → engine 層縫合 + 專屬單測。`boxes.id`/`keypoints.conf` 為 None → 哨兵值 + guard + 單測鎖死。Colab 斷線/VM 揮發 → Drive 持久化 + 全腳本 idempotent。官方站是小型大學伺服器 → 禮貌下載(retry/backoff/skip-existing),Kaggle 鏡像僅備援。小樣本波動 → 固定 seed 切分進版控 + 樣本數警語。Gradio 6 破壞性變更 → pin `>=6,<7`、不抄 5.x 範例。
