"""``fdp`` 命令列入口。

子命令(依 pipeline 順序):
  extract   影片(或目錄)→ keypoint cache parquet【唯一 GPU 步驟】
  detect    cache → events.json(純 CPU,毫秒級;--debug 另輸出特徵 JSONL)
  annotate  影片 + cache → 標註影片(H.264)+ events.json
  pipeline  extract → detect → annotate 一條龍(--source 0 可用 webcam 錄一段再處理)
  bench     影片 → FPS benchmark(可攜:任何機器都能補跑一列,不綁 Colab)

重依賴(torch/ultralytics/cv2)一律延遲到子命令內 import:
``fdp detect`` 在只裝核心依賴的環境(無 GPU)也能跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _write_debug_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_rules(cache_path: str, config_path: str, collect_debug: bool):
    from .config import load_config
    from .io.cache import read_cache
    from .rules import run_engine

    cfg = load_config(config_path)
    df, meta = read_cache(cache_path)
    events, debug = run_engine(df, meta.fps, cfg, collect_debug=collect_debug)
    return cfg, df, meta, events, debug


def cmd_extract(args: argparse.Namespace) -> int:
    """影片 → keypoint cache。--source 可為單一影片或裝滿影片的目錄。"""
    from .config import load_config
    from .inference.extract import extract_batch, extract_video

    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    src = Path(args.source)
    if src.is_dir():
        videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not videos:
            print(f"{src} 內沒有影片", file=sys.stderr)
            return 1
        extract_batch(videos, args.out, cfg, device=args.device)
    else:
        out = Path(args.out)
        out_path = out / f"{src.stem}.parquet" if out.suffix == "" else out
        meta = extract_video(src, out_path, cfg, device=args.device)
        print(f"cache → {out_path}({meta.n_frames} 幀,model={meta.model_name})")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """cache → events.json;--debug 另存 per-frame 特徵 JSONL(失敗分析用)。"""
    from .events.schema import write_events_json

    cfg, _, meta, events, debug = _run_rules(args.cache, args.config, args.debug is not None)
    write_events_json(args.out, events, source=meta.video_path, fps=meta.fps)
    if args.debug:
        _write_debug_jsonl(args.debug, debug)
    print(f"{Path(args.cache).stem}: {len(events)} 個事件 → {args.out}")
    for ev in events:
        print(
            f"  tracks={ev.track_ids} frames=[{ev.start_frame},{ev.end_frame}] "
            f"t=[{ev.start_time_s:.2f},{ev.end_time_s:.2f}]s rules={ev.rules_fired}"
        )
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    """影片 + cache → 標註影片(H.264);同時輸出 events.json。"""
    from .events.schema import write_events_json
    from .viz.annotate import annotate_video

    cfg, df, meta, events, debug = _run_rules(args.cache, args.config, True)
    out = annotate_video(args.video, df, meta.fps, cfg, events, debug, args.out)
    if args.events_out:
        write_events_json(args.events_out, events, source=str(args.video), fps=meta.fps)
    print(f"標註影片 → {out}({len(events)} 個事件)")
    return 0


def _record_webcam(index: int, duration_s: float, out_path: Path) -> Path:
    """webcam 錄一段到暫存影片(離線 pipeline 的 webcam 模式)。"""
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟 webcam {index}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = int(duration_s * fps)
    print(f"webcam 錄製 {duration_s:.0f}s({n} 幀)…")
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return out_path


def cmd_pipeline(args: argparse.Namespace) -> int:
    """extract → detect → annotate 一條龍,輸出標註影片與 events.json。"""
    import tempfile

    from .config import load_config
    from .events.schema import write_events_json
    from .inference.extract import extract_video
    from .rules import run_engine
    from .viz.annotate import annotate_video

    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model

    if str(args.source).isdigit():
        video = _record_webcam(
            int(args.source), args.duration, Path(tempfile.mkstemp(suffix=".mp4")[1])
        )
    else:
        video = Path(args.source)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{video.stem}.parquet"

    extract_video(video, cache_path, cfg, device=args.device)
    from .io.cache import read_cache

    df, meta = read_cache(cache_path)
    events, debug = run_engine(df, meta.fps, cfg, collect_debug=True)
    events_path = out_dir / f"{video.stem}.events.json"
    write_events_json(events_path, events, source=str(video), fps=meta.fps)
    if args.debug:
        _write_debug_jsonl(out_dir / f"{video.stem}.debug.jsonl", debug)
    annotated = annotate_video(
        video, df, meta.fps, cfg, events, debug, out_dir / f"{video.stem}_annotated.mp4"
    )
    print(f"完成:{annotated}、{events_path}({len(events)} 個事件)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """影片 → FPS benchmark(純推論 + 端到端 FPS、p50/p95 延遲)。

    可攜:任何機器都能對同一支(或任一支)影片補跑一列,不綁定 Colab——
    --model 可重複指定多次,一次跑完整個模型清單。
    """
    import platform

    from .bench.benchmark import benchmark, load_frames

    frames = load_frames(args.video, n_frames=args.n_frames)
    print(f"{args.video}: 載入 {len(frames)} 幀(要求 {args.n_frames})")

    results = []
    for model_name in args.model:
        r = benchmark(
            frames,
            model_name=model_name,
            device=args.device,
            quantize=args.quantize,
            n_runs=args.n_runs,
            warmup=args.warmup,
        )
        results.append(r.to_dict())
        print(
            f"  {model_name}: 純推論 {r.pure_inference_fps} FPS、端到端 {r.end_to_end_fps} FPS、"
            f"p50 {r.p50_latency_ms}ms、p95 {r.p95_latency_ms}ms"
        )

    if args.out:
        import torch
        import ultralytics

        payload = {
            "video": str(args.video),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "ultralytics_version": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "results": results,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="影片 → keypoint cache(GPU)")
    p.add_argument("--source", required=True, help="影片檔或影片目錄")
    p.add_argument("--out", required=True, help="輸出 parquet 或目錄")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", default=None, help="覆寫 config 的模型名(如 yolo26s-pose.pt)")
    p.add_argument("--device", default=None, help="cuda:0 / cpu(預設由 ultralytics 自選)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("detect", help="cache → events.json(純 CPU)")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True, help="events.json 輸出路徑")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--debug", default=None, help="per-frame 特徵 JSONL 輸出路徑")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("annotate", help="影片 + cache → 標註影片(H.264)")
    p.add_argument("--video", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True, help="標註影片輸出路徑")
    p.add_argument("--events-out", default=None, help="events.json 輸出路徑(可選)")
    p.add_argument("--config", default="config.yaml")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("pipeline", help="extract → detect → annotate 一條龍")
    p.add_argument("--source", required=True, help="影片檔,或 webcam 索引(如 0)")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--duration", type=float, default=10.0, help="webcam 錄製秒數")
    p.add_argument("--debug", action="store_true", help="輸出 per-frame 特徵 JSONL")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("bench", help="影片 → FPS benchmark(可攜,任何機器都能補跑)")
    p.add_argument("--video", required=True, help="固定用來計時的影片")
    p.add_argument("--model", action="append", required=True, help="模型名,可重複指定多次")
    p.add_argument("--device", default=None, help="cuda:0 / cpu(預設由 ultralytics 自選)")
    p.add_argument("--quantize", default=None, help="16 或 fp16 啟用 FP16 推論(僅 GPU 有意義)")
    p.add_argument("--n-frames", type=int, default=300, dest="n_frames")
    p.add_argument("--n-runs", type=int, default=3, dest="n_runs")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default=None, help="bench.json 輸出路徑(可選)")
    p.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
