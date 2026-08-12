"""Reproduce event-level evaluation from previously extracted keypoint caches."""

from __future__ import annotations

from pathlib import Path

from ..config import load_config
from ..io.cache import read_cache
from .ground_truth import load_gt_events
from .matching import evaluate_videos
from .report import build_video_dicts, list_failure_cases
from .splits import load_splits


def _compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_video"}


def _compact_failures(metrics: dict) -> dict:
    failures = list_failure_cases(metrics)
    return {
        "fp": [
            {"name": case["name"], "intervals": case["fp_intervals"]}
            for case in failures["fp_cases"]
        ],
        "fn": [
            {"name": case["name"], "intervals": case["fn_intervals"]}
            for case in failures["fn_cases"]
        ],
    }


def evaluate_cache_root(
    cache_root: str | Path,
    annotations_path: str | Path,
    splits_path: str | Path,
    config_path: str | Path,
    model_names: list[str],
    tol_s: float = 0.5,
) -> dict:
    """Evaluate frozen rules against the test split for one or more models.

    ``cache_root`` contains one directory per model, for example
    ``cache/yolo26n-pose/fall-01.parquet``. Pose inference stays outside this
    command so evaluation remains CPU-only and repeatable.
    """

    root = Path(cache_root)
    splits = load_splits(splits_path)
    cfg = load_config(config_path)
    gt_by_seq = load_gt_events(annotations_path)
    test_falls = list(splits["test"]["falls"])
    test_adls = list(splits["test"]["adls"])
    test_sequences = test_falls + test_adls
    adl_sequences = set(test_adls)

    models: dict[str, dict] = {}
    for model_name in model_names:
        cache_by_seq = {
            sequence: read_cache(root / model_name / f"{sequence}.parquet")
            for sequence in test_sequences
        }
        videos = build_video_dicts(
            cache_by_seq,
            test_sequences,
            adl_sequences,
            gt_by_seq,
            cfg,
        )
        metrics = evaluate_videos(videos, tol_s=tol_s)
        models[model_name] = {
            "metrics": _compact_metrics(metrics),
            "failure_cases": _compact_failures(metrics),
        }

    return {
        "protocol": {
            "split": "test",
            "tol_s": tol_s,
            "n_falls": len(test_falls),
            "n_adls": len(test_adls),
        },
        "models": models,
    }
