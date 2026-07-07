# fall-detection-pose

> 🚧 v1 開發中 — 依 [plan.md](plan.md) 實作,README 完整版(架構圖、評估表、失敗分析)於 M7 里程碑定稿。

以 **YOLO26-pose 預訓練模型 + ByteTrack 多目標追蹤**為基礎的規則式(rule-based)跌倒偵測系統。
本專案重點不在模型創新,而在工程能力:

- **可解釋的規則引擎**:每個 track 一台狀態機(UPRIGHT → FALLING → FALLEN → ALARM),
  所有閾值集中於 [config.yaml](config.yaml),每個值附選擇理由與文獻出處。
- **event-level 誠實評估**:在 UR Fall Detection Dataset(30 falls + 40 ADL)上以
  明確定義的事件配對協定計算 precision / recall / F1;tune/test 切分防止「在測試集上調參」。
- **失敗分析**:對誤報與漏報案例附特徵時序圖,展示規則「為什麼」觸發或錯過。
- **推論與規則解耦**:GPU 只跑一次姿態抽取落成 keypoint cache,調參/評估為秒級 CPU 工作。

## 狀態

| 里程碑 | 內容 | 狀態 |
|---|---|---|
| M0 | 專案骨架 | ✅ |
| M1 | 規則引擎 + 合成軌跡單元測試 | ✅ |
| M2 | 推論 pipeline + 煙測 notebook | ✅ |
| M3 | URFD 全量抽取 | ✅ |
| M4 | 閾值校準 + 評估 + 失敗分析 | ✅ |
| M5 | FPS benchmark | 🔄 |
| M6 | Gradio demo | ⬜ |
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
