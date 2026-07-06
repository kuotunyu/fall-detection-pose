"""Keypoint cache 的 roundtrip 與 schema 驗證測試。"""

import numpy as np
import pytest

from fall_detection.io.cache import (
    SCHEMA_VERSION,
    CacheMeta,
    CacheSchemaError,
    read_cache,
    write_cache,
)

from synthetic import make_trajectory


def _meta(**overrides) -> CacheMeta:
    base = dict(
        schema_version=SCHEMA_VERSION,
        video_path="dummy.mp4",
        video_sha1="0" * 40,
        fps=30.0,
        width=640,
        height=480,
        n_frames=168,
        model_name="yolo26n-pose.pt",
        ultralytics_version="8.4.x",
        tracker_yaml="bytetrack.yaml",
        conf=0.25,
        iou=0.5,
        device="cuda:0",
        git_commit="abc1234",
    )
    base.update(overrides)
    return CacheMeta(**base)


def test_roundtrip_values_and_meta(tmp_path):
    df = make_trajectory([("stand", 1.0), ("to:lie", 0.6), ("lie", 1.0)], fps=30.0)
    path = tmp_path / "clip.parquet"
    write_cache(df, _meta(), path)
    assert path.exists()
    assert path.with_suffix(".parquet.meta.json").exists()

    df2, meta2 = read_cache(path)
    assert len(df2) == len(df)
    assert meta2.fps == 30.0
    assert meta2.model_name == "yolo26n-pose.pt"
    np.testing.assert_allclose(
        np.stack(df2["kpts_xy"].to_numpy()),
        np.stack(df["kpts_xy"].to_numpy()),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        df2["frame_idx"].to_numpy(), df["frame_idx"].to_numpy()
    )


def test_schema_version_mismatch_raises(tmp_path):
    df = make_trajectory([("stand", 0.5)], fps=30.0)
    path = tmp_path / "old.parquet"
    write_cache(df, _meta(schema_version=999), path)
    with pytest.raises(CacheSchemaError):
        read_cache(path)


def test_roundtrip_survives_engine(tmp_path, cfg):
    """寫入→讀回的資料餵引擎,結果與原始 DataFrame 一致。"""
    from fall_detection.rules import run_engine

    df = make_trajectory([("stand", 2.0), ("to:lie", 0.6), ("lie", 3.0)], fps=30.0)
    path = tmp_path / "clip.parquet"
    write_cache(df, _meta(), path)
    df2, meta2 = read_cache(path)
    ev_a, _ = run_engine(df, 30.0, cfg)
    ev_b, _ = run_engine(df2, meta2.fps, cfg)
    assert [(e.start_frame, e.end_frame) for e in ev_a] == [
        (e.start_frame, e.end_frame) for e in ev_b
    ]
