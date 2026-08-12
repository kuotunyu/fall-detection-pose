# Third-party notices

## UR Fall Detection Dataset (URFD)

本專案的評估資料來自 Kwolek 與 Kepski 建立的
[UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)。
資料集及其衍生展示媒體依原作者的 **CC BY-NC-SA 4.0** 條款使用，
不屬於本 repository 的 MIT License 授權範圍。

本 repository 不重新散布原始資料集。以下展示素材由 URFD 樣本產生，
僅用於非商業研究與作品集展示：

- `assets/demo_fall.gif`
- `assets/demo_adl.png`
- `assets/demo_mobile.png`

資料集引用：

> B. Kwolek and M. Kepski, "Human fall detection on embedded platform using
> depth maps and wireless accelerometer," *Computer Methods and Programs in
> Biomedicine*, 2014. <https://doi.org/10.1016/j.cmpb.2014.09.005>

## Ultralytics 與 YOLO 權重

本專案的 `infer` extra 會安裝 [Ultralytics](https://github.com/ultralytics/ultralytics)，
並於首次推論時下載官方 YOLO26-pose 預訓練權重。本 repository 不重新散布
Ultralytics 套件或模型權重。

依 [Ultralytics 官方授權說明](https://www.ultralytics.com/license)，其開源軟體及
官方訓練模型預設適用 **AGPL-3.0**；若要整合至商業或封閉原始碼產品，使用者應
另行確認 Ultralytics Enterprise License 等適用條款。這些第三方元件不屬於本
repository 的 MIT License 授權範圍。

## 本專案原始程式碼

本 repository 由專案作者撰寫的原始程式碼依 [MIT License](LICENSE) 釋出。
所有第三方套件、模型權重、資料集與衍生媒體仍各自適用其原始授權條款。
