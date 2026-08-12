"""Synchronize the Gradio demo notebook with the current local interface."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/05_gradio_demo.ipynb")

INTRODUCTION = """# 05 — Gradio Demo

**用途**：在 Colab 啟動與本機相同的四狀態介面，分析流程為：

1. 選擇短片。
2. 依序執行影片解碼、Pose extraction、Event detection 與 Video annotation。
3. 在同一工作區檢視標註影片、事件區間、Track ID、實際 `Rules fired` 與
   `events.json`。無事件時會顯示明確的 0 事件結論。

本機使用者可直接執行 README 的 `uv run python -m fall_detection.app.gradio_app
--no-share`，不需要先操作本 notebook。本 notebook 保留作為 Colab 重現路徑。

**執行方式**：`Runtime → Run all`（GPU runtime，T4 即可；CPU 也能執行但較慢）。
最後一格會持續執行，因為 `gr.Blocks.launch()` 正在提供網頁服務；停止時手動中斷該格。

範例使用 `fall-06` 與 `adl-01`，分別對應 README 中的實際 ALARM 與 0 事件媒體。
"""

EXAMPLE_CELL = """from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/fall-detection-pose'
DATA_DIR = f'{DRIVE_ROOT}/data/urfd'
VIDEO_DIR = f'{DATA_DIR}/videos'

# 與 README 的實際 pipeline 媒體一致：fall-06 形成 ALARM；adl-01 為 0 事件。
# 找不到時保留空範例區，使用者仍可自行上傳影片。
example_candidates = [f'{VIDEO_DIR}/fall-06.mp4', f'{VIDEO_DIR}/adl-01.mp4']
examples = [p for p in example_candidates if os.path.exists(p)]
missing = [p for p in example_candidates if p not in examples]
if missing:
    print(f'警告：找不到範例影片 {missing}（notebook 02 是否已跑過？），仍可手動上傳')
print('範例影片：', examples)
"""

OUTRO = """## 結果與 README 媒體

README 的 `assets/demo_fall.gif`、`assets/demo_adl.png` 與 `assets/demo_mobile.png`
由本專案的 Playwright 擷取腳本從實際 pipeline 結果產生。若介面改版，請在本機重新執行
`scripts/capture_demo_media.py`，不要手動拼接推論數值或修改事件結果。
"""


def _lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = _lines(INTRODUCTION)
    notebook["cells"][4]["source"] = _lines(EXAMPLE_CELL)
    notebook["cells"][6]["source"] = _lines(OUTRO)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
