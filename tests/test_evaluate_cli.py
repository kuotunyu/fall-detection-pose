import json

from synthetic import make_trajectory

from fall_detection.cli import main
from fall_detection.io.cache import SCHEMA_VERSION, CacheMeta, write_cache


def _write_cache(path, sequence, model_name, trajectory):
    write_cache(
        trajectory,
        CacheMeta(
            schema_version=SCHEMA_VERSION,
            video_path=f"{sequence}.mp4",
            video_sha1="0" * 40,
            fps=30.0,
            width=640,
            height=480,
            n_frames=len(trajectory),
            model_name=f"{model_name}.pt",
            ultralytics_version="8.4.x",
            tracker_yaml="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            device="cpu",
        ),
        path,
    )


def test_evaluate_command_reproduces_metrics_from_real_caches(tmp_path):
    """Removing cache loading, GT conversion, or event matching must fail."""
    model_name = "yolo26n-pose"
    model_dir = tmp_path / "cache" / model_name
    _write_cache(
        model_dir / "fall-01.parquet",
        "fall-01",
        model_name,
        make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=30.0),
    )
    _write_cache(
        model_dir / "adl-01.parquet",
        "adl-01",
        model_name,
        make_trajectory([("walk", 5.0)], fps=30.0, seed=1),
    )

    annotations = tmp_path / "falls.csv"
    annotations.write_text(
        "fall-01,60,0\nfall-01,78,1\nfall-01,165,1\n", encoding="utf-8"
    )
    splits = tmp_path / "splits.yaml"
    splits.write_text(
        "seed: 42\n"
        "tune:\n  falls: []\n  adls: []\n"
        "test:\n  falls: [fall-01]\n  adls: [adl-01]\n",
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"

    try:
        return_code = main(
            [
                "evaluate",
                "--cache-root",
                str(tmp_path / "cache"),
                "--annotations",
                str(annotations),
                "--splits",
                str(splits),
                "--config",
                "config.yaml",
                "--model",
                model_name,
                "--out",
                str(output),
            ]
        )
    except SystemExit as exc:
        return_code = int(exc.code)

    assert return_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["protocol"] == {
        "split": "test",
        "tol_s": 0.5,
        "n_falls": 1,
        "n_adls": 1,
    }
    assert report["models"][model_name]["metrics"] == {
        "tol_s": 0.5,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "video_level_specificity": 1.0,
        "n_videos": 2,
        "n_adl_videos": 1,
    }
    assert report["models"][model_name]["failure_cases"] == {
        "fp": [],
        "fn": [],
    }
