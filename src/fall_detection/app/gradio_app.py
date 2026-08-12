"""Gradio 6 demo:上傳影片 → 標註影片 + 事件表 + events.json 下載。

模組頂層刻意不 import gradio/torch/ultralytics/cv2(同 cli.py 的原則):
``_events_to_rows`` 之類純函式因此能在無 GPU/無 gradio 的本機輕量 venv 被
匯入與單元測試;``gr.Progress()`` 需要在函式簽名的預設值就是一個 gradio
物件才能被辨識,因此改用 ``build_demo`` 內的 closure 包一層,而不是讓
``process_video`` 本身依賴 gradio。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable, Iterator

DEFAULT_MODEL_CHOICES = ["yolo26n-pose.pt", "yolo26s-pose.pt"]


@dataclass(frozen=True)
class StreamMessage:
    """A progress or terminal message emitted by the background analysis."""

    kind: str
    fraction: float = 0.0
    description: str = ""
    annotated_path: str | None = None
    events_path: str | None = None
    error: Exception | None = None
EVENT_TABLE_HEADERS = ["Track ID", "開始時間(s)", "結束時間(s)", "時長(s)", "觸發規則"]


def _events_to_rows(events: list[dict]) -> list[list]:
    """``events.json`` 的 ``events`` 陣列(或 ``FallEvent.to_dict()`` 列表)→
    ``gr.Dataframe`` 要的 list-of-rows。"""
    return [
        [
            ",".join(str(t) for t in e["track_ids"]),
            round(e["start_time_s"], 2),
            round(e["end_time_s"], 2),
            round(e["duration_s"], 2),
            ", ".join(e["rules_fired"]),
        ]
        for e in events
    ]


def _analysis_metadata(df, meta) -> dict[str, int]:
    """Return stable analysis counts without counting untracked detections."""

    if df.empty or "track_id" not in df:
        n_tracks = 0
    else:
        valid_tracks = df.loc[df["track_id"] >= 0, "track_id"]
        n_tracks = int(valid_tracks.nunique())
    return {"n_frames": int(meta.n_frames), "n_tracks": n_tracks}


def _stream_process(
    video_path: str,
    model_name: str,
    config_path: str,
    *,
    runner: Callable | None = None,
) -> Iterator[StreamMessage]:
    """Run inference off the UI thread and yield its progress messages."""

    selected_runner = runner or process_video
    messages: Queue[StreamMessage] = Queue()

    def on_progress(fraction: float, description: str) -> None:
        messages.put(
            StreamMessage(
                kind="progress",
                fraction=max(0.0, min(float(fraction), 1.0)),
                description=str(description),
            )
        )

    def worker() -> None:
        try:
            annotated_path, _rows, events_path = selected_runner(
                video_path,
                model_name,
                config_path,
                on_progress=on_progress,
            )
            messages.put(
                StreamMessage(
                    kind="success",
                    fraction=1.0,
                    description="分析完成",
                    annotated_path=annotated_path,
                    events_path=events_path,
                )
            )
        except Exception as exc:  # noqa: BLE001 - forwarded to the safe UI mapper
            messages.put(StreamMessage(kind="error", error=exc))

    Thread(target=worker, daemon=True).start()
    while True:
        message = messages.get()
        yield message
        if message.kind in {"success", "error"}:
            return


def process_video(
    video_path: str,
    model_name: str,
    config_path: str = "config.yaml",
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[str, list[list], str]:
    """上傳影片 → (標註影片路徑, 事件表格 rows, events.json 路徑)。

    固定用同一個工作目錄(每次呼叫覆寫上一次的輸出),不是每次呼叫都開新的
    ``mkdtemp``:demo 的 ``concurrency_limit=1`` 保證不會有兩個請求同時寫入,
    這在 Colab 這種一次性 session 裡無關緊要,但這個 app 也會部署成長時間跑的
    HF Space,每次呼叫都留一份新暫存檔會讓磁碟用量無界成長。
    """
    if not video_path:
        raise ValueError("請先上傳影片")

    def _progress(frac: float, desc: str) -> None:
        if on_progress is not None:
            on_progress(frac, desc)

    from ..config import load_config
    from ..events.schema import write_events_json
    from ..inference.extract import extract_video
    from ..io.cache import read_cache
    from ..rules import run_engine
    from ..viz.annotate import annotate_video

    cfg = load_config(config_path)
    if model_name:
        cfg.model.name = model_name

    work_dir = Path(tempfile.gettempdir()) / "fdp_demo_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_path = work_dir / "cache.parquet"
    annotated_path = work_dir / "annotated.mp4"
    events_path = work_dir / "events.json"

    _progress(0.02, "載入模型、姿態抽取中…")
    extract_video(
        Path(video_path),
        cache_path,
        cfg,
        on_frame=lambda i, n: _progress(
            0.05 + 0.65 * (i + 1) / max(n, 1), f"姿態抽取中 {i + 1}/{n}"
        ),
    )

    _progress(0.72, "規則引擎判定中…")
    df, meta = read_cache(cache_path)
    events, debug = run_engine(df, meta.fps, cfg, collect_debug=True)

    _progress(0.82, "輸出標註影片(H.264 重編碼)…")
    annotate_video(video_path, df, meta.fps, cfg, events, debug, annotated_path)

    write_events_json(
        events_path,
        events,
        source=str(video_path),
        fps=meta.fps,
        extra=_analysis_metadata(df, meta),
    )

    _progress(1.0, "完成")
    rows = _events_to_rows([e.to_dict() for e in events])
    return str(annotated_path), rows, str(events_path)


def build_demo(config_path: str = "config.yaml", example_videos: list[str] | None = None):
    """組出 ``gr.Blocks`` demo(不呼叫 ``launch()``,方便 notebook/測試各自決定)。"""
    import gradio as gr

    with gr.Blocks(title="跌倒偵測 Demo") as demo:
        gr.Markdown(
            "# 跌倒偵測 Demo\n"
            "YOLO26-pose + ByteTrack + 規則式狀態機(不訓練模型,純規則引擎判定)。"
            "上傳一支影片,輸出標註影片(骨架 + track id + 狀態 + ALARM 橫幅)與"
            "偵測到的跌倒事件表。首次執行需下載模型權重,請稍候。\n\n"
            "評估協定、閾值理由與失敗案例分析均收錄於專案 README。"
        )
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(sources=["upload"], label="上傳影片")
                model_in = gr.Dropdown(
                    DEFAULT_MODEL_CHOICES,
                    value=DEFAULT_MODEL_CHOICES[0],
                    label="模型(n=快、s=較準,見 README 評估表)",
                )
                run_btn = gr.Button("開始偵測", variant="primary")
                if example_videos:
                    gr.Examples(examples=[[p] for p in example_videos], inputs=[video_in])
            with gr.Column():
                video_out = gr.Video(label="標註結果")
                table_out = gr.Dataframe(headers=EVENT_TABLE_HEADERS, label="偵測到的跌倒事件")
                file_out = gr.File(label="下載 events.json")

        def _handler(video_path, model_name, progress=gr.Progress()):
            def on_progress(frac: float, desc: str) -> None:
                progress(frac, desc=desc)

            try:
                return process_video(video_path, model_name, config_path, on_progress=on_progress)
            except Exception as e:  # noqa: BLE001 - demo 是 share=True 公開端點,任何輸入都不能讓伺服器整個炸掉
                raise gr.Error(f"處理失敗:{e}") from e

        run_btn.click(
            fn=_handler,
            inputs=[video_in, model_in],
            outputs=[video_out, table_out, file_out],
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    """獨立啟動(``python -m fall_detection.app.gradio_app``);notebook 05 直接呼叫
    ``build_demo`` 較方便帶入 Drive 上的範例影片路徑,不走這個入口。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--examples", nargs="*", default=None, help="範例影片路徑(可選,可多個)")
    parser.add_argument("--no-share", action="store_true", help="停用 public link(僅本機測試用)")
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()

    demo = build_demo(config_path=args.config, example_videos=args.examples)
    demo.queue().launch(
        share=not args.no_share, max_file_size="200mb", server_port=args.server_port
    )


if __name__ == "__main__":
    main()
