# fall-detection-pose

> 🚧 v1 開發中 — 依 [plan.md](plan.md) 實作,僅架構圖與最終潤稿待 M7 里程碑定稿。

以 **YOLO26-pose 預訓練模型 + ByteTrack 多目標追蹤**為基礎的規則式(rule-based)跌倒偵測系統。
本專案重點不在模型創新,而在工程能力:

- **可解釋的規則引擎**:每個 track 一台狀態機(UPRIGHT → FALLING → FALLEN → ALARM),
  所有閾值集中於 [config.yaml](config.yaml),每個值附選擇理由與文獻出處。
- **event-level 誠實評估**:在 UR Fall Detection Dataset(30 falls + 40 ADL)上以
  明確定義的事件配對協定計算 precision / recall / F1;tune/test 切分防止「在測試集上調參」。
- **失敗分析**:對誤報與漏報案例附特徵時序圖,展示規則「為什麼」觸發或錯過。
- **推論與規則解耦**:GPU 只跑一次姿態抽取落成 keypoint cache,調參/評估為秒級 CPU 工作。

## Demo

<table>
<tr>
<td align="center"><b>跌倒 → 正確觸發 ALARM</b></td>
<td align="center"><b>日常動作(ADL)→ 正確不觸發</b></td>
</tr>
<tr>
<td><img src="assets/demo_fall.gif" width="400"></td>
<td><img src="assets/demo_adl.gif" width="400"></td>
</tr>
</table>

兩支都是實際跑完整條 pipeline 的畫面(非後製剪輯),用
[notebooks/05_gradio_demo.ipynb](notebooks/05_gradio_demo.ipynb) 啟動的 Gradio demo 直接
螢幕錄影:`fall-06`(tune split)展示跌倒正確觸發紅色 ALARM 橫幅;`adl-01`(test split)
展示日常動作(過程中甚至有蹲下這種容易與跌倒混淆的姿勢)全程正確保持 UPRIGHT、不誤觸。

## 狀態

| 里程碑 | 內容 | 狀態 |
|---|---|---|
| M0 | 專案骨架 | ✅ |
| M1 | 規則引擎 + 合成軌跡單元測試 | ✅ |
| M2 | 推論 pipeline + 煙測 notebook | ✅ |
| M3 | URFD 全量抽取 | ✅ |
| M4 | 閾值校準 + 評估 + 失敗分析 | ✅ |
| M5 | FPS benchmark | ✅ |
| M6 | Gradio demo | ✅ |
| M7 | README 定稿 | ⬜ |

## 評估

協定:event-level 配對,預測與 GT ± 0.5s 容忍窗內有任何時間交集即視為候選,
每個 GT 貪婪配對交集最大的一個預測(一對一)。ADL 影片一律視為 0 個跌倒 GT
——即使該影片的姿態標註本身出現「躺姿」區間(URFD 的 ADL 集合刻意包含主動
躺下,如躺床上,用來測試誤報率),任何預測事件都算 FP。閾值全部在 tune split
(10 falls + 13 adls)網格搜尋校準凍結(見 [config.yaml](config.yaml));
以下為 **test split**(20 falls + 27 adls,從未參與調參)的結果,原始數字見
[eval/metrics.json](eval/metrics.json)。

| 模型 | Precision | Recall | F1 | Video-level Specificity |
|---|---|---|---|---|
| yolo26n-pose(預設) | 0.600 | 0.600 | 0.600 | 0.741 |
| yolo26s-pose | 0.611 | 0.550 | 0.579 | 0.778 |

調參經過兩輪:第一輪勝出的 4 個閾值全部卡在候選範圍邊緣(方法論警訊,代表
範圍切太窄);往更敏感方向擴大網格、並修掉一個結構性 bug(track 消失時最後
觀測已符合躺姿卻沒收尾成事件)後,第二輪 F1(yolo26n-pose)從 0.457 提升到
0.600。調參準則:recall 優先、precision ≥ 0.5 才列入候選——跌倒漏判(沒人去
查看)的代價高於一次誤報。

## Benchmark

固定一支 URFD 重組影片(`adl-01.mp4`,150 幀,實際可用幀數誠實回報而非湊滿
300)先整支解碼進記憶體,`{yolo26n-pose, yolo26s-pose} x {GPU FP32, GPU FP16,
CPU}` 各跑 3 輪取中位數;GPU 計時前後夾 `torch.cuda.synchronize()`,CPU 為
Colab 標準 2 vCPU(數字偏低,誠實照報)。**不引用官方「CPU 快 43%」的宣傳
數字**——那是 detect 模型的 ONNX 匯出數字,不適用本專案的 pose 模型。原始
數字見 [bench.json](bench.json)。

| 模型 | 裝置 | 端到端 FPS | p50 延遲 | p95 延遲 |
|---|---|---|---|---|
| yolo26n-pose | GPU(T4)FP32 | 59.65 | 13.84ms | 23.52ms |
| yolo26n-pose | GPU(T4)FP16 | 64.64 | 15.63ms | 24.25ms |
| yolo26n-pose | CPU(2 vCPU) | 8.23 | 116.96ms | 178.96ms |
| yolo26s-pose | GPU(T4)FP32 | 72.25 | 13.88ms | 20.40ms |
| yolo26s-pose | GPU(T4)FP16 | 66.18 | 14.69ms | 24.09ms |
| yolo26s-pose | CPU(2 vCPU) | 3.36 | 271.15ms | 408.98ms |

兩個模型在 GPU 上都遠超即時(30fps)所需;CPU 上 `yolo26n-pose` 仍有 8+ FPS
堪用,`yolo26s-pose` 掉到 3.36 FPS 明顯吃緊。GPU 對較大模型的加速幅度也更大
(n:~7.3x、s:~21.5x),符合計算量較重的模型從平行化得利更多的預期。

**測出來的兩個反直覺數字,誠實報告、不隱藏**:
- `yolo26s-pose` 在 GPU 上量到比 `yolo26n-pose` 更快(73.53 vs 60.65 FPS
  純推論)。CPU 上兩者關係符合預期(s 比 n 慢 ~2.45 倍,計算量差異的合理
  反映),因此 GPU 這個反轉不太像是程式呼叫錯模型的 bug,較可能是量測順序
  效應(n 先跑,GPU/CUDA kernel 尚未完全暖機)或 Colab 共用 T4 的量測雜訊
  ——樣本數(150 幀 x 3 輪)不足以下更強的結論。
- FP16 對 `yolo26n-pose` 有加速(+8.3%),對 `yolo26s-pose` 反而略慢(-8.6%),
  同樣可能是雜訊,也可能反映小模型的 FP16 casting 額外開銷相對計算量比例
  較大,抵銷部分理論加速。
- benchmark 腳本(`fdp bench` 或 `src/fall_detection/bench/benchmark.py`)
  可攜,任何機器都能補跑一列驗證,不綁定本次 Colab session 的結果。

## 失敗分析

從 test split 挑 1 個漏報(FN)+ 2 個誤報(FP),展示規則引擎「為什麼」錯過或
誤觸發(特徵時序圖見 `notebooks/03_tune_eval.ipynb` 失敗案例分析格):

**FN — `fall-21`**:追蹤器在跌倒的視覺證據真正成形前就整個失去目標。全程軀幹
傾角(θ)只有 0-6°、bbox 寬高比只有 ~0.3,沒有任何躺姿跡象;垂直速度確實
持續爬升,但直到資料結束前才剛好接近門檻,track 就消失了。這是**追蹤持續度**
的限制,不是閾值問題——沒有更多資料,任何規則法都生不出證據。

**FP — `adl-34`**:URFD 的 ADL 集合刻意包含「主動躺下」的日常動作(測試系統
會不會把臥床誤判成跌倒)。這支影片裡 θ 在 6 秒多內反覆於躺姿(~90°)與非躺姿
(~20-30°)之間震盪多次,是「躺下→坐起→躺下…」的主動調整,而非一次性站立→
跌倒→臥地不動。系統從純幾何角度正確判斷「持續符合躺姿」,但協定規定 ADL 影片
零 GT——這是規則法在「主動躺下 vs. 跌倒臥地」上幾何不可分辨的已知極限。
(附帶一提:雖然姿態震盪劇烈,因全程同一個 track id,靠回正需要「持續」的
遲滯設計撐住,並沒有被切成好幾段事件——跟下一個案例形成對比。)

**FP — `fall-08`**:這支**其實是真實跌倒**,但被切成兩個 track id(ByteTrack
在跌倒過程一次短暫的姿態回彈時斷了 id)。兩段預測分別是 `[0.933s,2.233s]`
(track 1,靠 finalize-while-FALLEN 修正正確收尾,對到 GT 算 TP)與
`[2.367s,3.0s]`(track 2,全新 id 重新觸發,判定為「重複預測」算 FP)。根本
原因是**縫合機制與事件合併機制之間的縫隙**:track 縫合靠 bbox IoU + 時間視窗
判斷是否同一人,這次因短暫回彈使 bbox 形狀變化過大沒縫上;事件合併只看
track id 鏈是否有交集,縫合一旦失敗就無法回頭補救,即使兩段事件時間相鄰、
位置相同。

## 已知限制與未來工作

- **`theta_lying_enter=40°` 低於文獻安全下限**:tune split 網格調參結果比
  Chen et al. 建議的 45° 更低(該研究指出 45° 會誤判深彎腰)。這 70 支 URFD
  影片沒有彎腰類 ADL 可測出此風險,是刻意接受的類外泛化風險——換到有彎腰/
  伸展動作的場域需重新校準。
- **track 縫合/事件合併縫隙**(見上方 `fall-08` 分析):跨 track id 鏈的事件
  合併目前只認 id 交集,不認時間相鄰 + 空間鄰近。
- **收尾事件不一定會顯示即時 ALARM 畫面**:`track_lost_while_fallen` 這類收尾
  規則能正確記錄事件,但如果一次跌倒過程 track id 反覆中斷(如 `fall-01`:
  1→3→5 換了三次),「持續躺姿」的計時會跟著中斷重算,狀態機永遠撐不到即時
  的 FALLEN→ALARM 轉換,只能靠影片結束時的收尾機制補判——事件本身正確,但
  標註影片上看不到紅色 ALARM 橫幅。demo GIF 因此選了 track id 全程穩定的
  `fall-06`,而非最早測試、更能體現縫合機制韌性的 `fall-01`。
- **慢速跌倒是規則法已知盲區**(見 `tests/test_engine.py` 的
  `test_slow_lie_down_no_event`):速度/角速度是進入 FALLING 的必要條件,
  刻意緩慢的跌倒或躺下不會觸發。
- **`model.conf` 無法事後調參**:偵測信心門檻在 GPU extract 階段就烘進
  keypoint cache,調整需要重新抽取,不像規則引擎閾值能在 CPU 上秒級迭代。

## 資料集聲明

評估使用 [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)
(CC BY-NC-SA 4.0,**不隨本 repo 散布**,由下載腳本自官方站取得):

> Bogdan Kwolek, Michal Kepski, "Human fall detection on embedded platform using
> depth maps and wireless accelerometer", *Computer Methods and Programs in
> Biomedicine*, Vol. 117, Issue 3, 2014, pp. 489–501.
> DOI: [10.1016/j.cmpb.2014.09.005](https://doi.org/10.1016/j.cmpb.2014.09.005)

## License

程式碼採 [MIT](LICENSE);資料集授權見上。
