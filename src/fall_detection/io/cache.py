"""Keypoint cache:推論(GPU)與規則引擎(CPU)之間的交接格式。

一支影片一個 parquet(每列 = 一個 (frame, track) 偵測)+ 同名 ``.meta.json``。
中繼資料同時冗餘寫入 parquet 的 schema metadata,防止單一檔案遺失。
``schema_version`` 嚴格比對:不相容直接拋 :class:`CacheSchemaError`,
提示重跑 extract,不做任何隱式降級——快取與程式版本漂移是評估數字失真的
最大來源,寧可 fail-fast。

哨兵值約定(對應 ultralytics 回傳 None 的情況):
- ``track_id = -1``:該偵測未被指派 track(``boxes.id is None``);
- ``kpts_conf`` 整列 ``-1.0``:模型未輸出 keypoint 置信度
  (``keypoints.conf is None``),下游一律視為不可信。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
N_KPTS = 17

_META_KEY = b"fdp_meta"

ARROW_SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int32()),
        ("t_ms", pa.float64()),
        ("track_id", pa.int32()),
        ("bbox_x1", pa.float32()),
        ("bbox_y1", pa.float32()),
        ("bbox_x2", pa.float32()),
        ("bbox_y2", pa.float32()),
        ("bbox_conf", pa.float32()),
        ("kpts_xy", pa.list_(pa.float32(), 2 * N_KPTS)),
        ("kpts_conf", pa.list_(pa.float32(), N_KPTS)),
    ]
)

CACHE_COLUMNS = [f.name for f in ARROW_SCHEMA]


class CacheSchemaError(RuntimeError):
    """cache 的 schema_version 與目前程式不相容。"""


@dataclass
class CacheMeta:
    """cache 的完整出處紀錄,足以判斷「這份快取是怎麼來的」。"""

    schema_version: int
    video_path: str
    video_sha1: str
    fps: float
    width: int
    height: int
    n_frames: int
    model_name: str
    ultralytics_version: str
    tracker_yaml: str
    conf: float
    iou: float
    device: str
    git_commit: str = ""


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def write_cache(df: pd.DataFrame, meta: CacheMeta, path: str | Path) -> None:
    """寫出 parquet + sidecar meta json(目錄不存在時自動建立)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        df[CACHE_COLUMNS], schema=ARROW_SCHEMA, preserve_index=False
    )
    meta_json = json.dumps(asdict(meta), ensure_ascii=False)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, _META_KEY: meta_json.encode()})
    pq.write_table(table, path)
    _sidecar_path(path).write_text(meta_json, encoding="utf-8")


def read_cache(path: str | Path) -> tuple[pd.DataFrame, CacheMeta]:
    """讀取 cache 並驗證 schema_version;回傳 (rows, meta)。

    meta 優先取 parquet 內嵌版本,缺失時退回 sidecar json;兩者皆無視為損毀。
    """
    path = Path(path)
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(_META_KEY)
    if raw is None:
        sidecar = _sidecar_path(path)
        if not sidecar.exists():
            raise CacheSchemaError(f"{path} 缺少中繼資料(parquet metadata 與 sidecar 皆無)")
        raw = sidecar.read_text(encoding="utf-8")
    meta = CacheMeta(**json.loads(raw))
    if meta.schema_version != SCHEMA_VERSION:
        raise CacheSchemaError(
            f"cache schema_version={meta.schema_version} 與程式 SCHEMA_VERSION={SCHEMA_VERSION} "
            f"不相容:請以目前版本重跑 `fdp extract`(檔案:{path})"
        )
    return table.to_pandas(), meta
