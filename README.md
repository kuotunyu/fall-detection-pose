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
| M3 | URFD 全量抽取 | 🔄 |
| M4 | 閾值校準 + 評估 + 失敗分析 | ⬜ |
| M5 | FPS benchmark | ⬜ |
| M6 | Gradio demo | ⬜ |
| M7 | README 定稿 | ⬜ |

## 資料集聲明

評估使用 [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)
(CC BY-NC-SA 4.0,**不隨本 repo 散布**,由下載腳本自官方站取得):

> Bogdan Kwolek, Michal Kepski, "Human fall detection on embedded platform using
> depth maps and wireless accelerometer", *Computer Methods and Programs in
> Biomedicine*, Vol. 117, Issue 3, 2014, pp. 489–501.
> DOI: [10.1016/j.cmpb.2014.09.005](https://doi.org/10.1016/j.cmpb.2014.09.005)

## License

程式碼採 [MIT](LICENSE);資料集授權見上。
