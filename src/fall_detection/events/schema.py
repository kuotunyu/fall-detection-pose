"""跌倒事件的資料結構、後處理與 events.json 序列化。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import EventsConfig


@dataclass
class FallEvent:
    """一次跌倒事件。

    ``track_ids`` 為串列:track 縫合(ByteTrack 在跌倒瞬間斷 id)時,
    同一事件可能橫跨多個 track id。``peak_features`` 與 ``rules_fired``
    記錄觸發當下的證據,供除錯、失敗分析與 demo 顯示。
    """

    track_ids: list[int]
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    peak_features: dict = field(default_factory=dict)
    rules_fired: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.duration_s, 3)
        return d


def postprocess_events(events: list[FallEvent], cfg: EventsConfig) -> list[FallEvent]:
    """事件後處理:先合併、再濾短。

    合併:同一 track 鏈(track_ids 有交集)且時間間隔 < ``merge_gap_s`` 的相鄰
    事件視為同一次跌倒被 keypoint dropout 切開,合併之。
    濾短:短於 ``min_event_duration_s`` 的事件為抖動殘渣,直接剔除。
    順序不可反:先濾短可能把「半段真事件」丟掉,合併後就救不回來。
    """
    if not events:
        return []
    events = sorted(events, key=lambda e: e.start_time_s)
    merged: list[FallEvent] = []
    for ev in events:
        # 往回找「最近一個同 track 鏈」的事件:多人場景下,別人的事件可能
        # 插在同一鏈的兩個片段之間,只比對 merged[-1] 會擋住應發生的合併
        target = None
        for cand in reversed(merged):
            if set(cand.track_ids) & set(ev.track_ids):
                target = cand
                break
        if target is not None and ev.start_time_s - target.end_time_s < cfg.merge_gap_s:
            target.end_frame = max(target.end_frame, ev.end_frame)
            target.end_time_s = max(target.end_time_s, ev.end_time_s)
            target.track_ids = sorted(set(target.track_ids) | set(ev.track_ids))
            target.rules_fired = sorted(set(target.rules_fired) | set(ev.rules_fired))
            for k, v in ev.peak_features.items():
                target.peak_features[k] = max(
                    target.peak_features.get(k, float("-inf")), v
                )
        else:
            merged.append(ev)
    return [e for e in merged if e.duration_s >= cfg.min_event_duration_s]


def events_to_json_dict(
    events: list[FallEvent], source: str, fps: float, extra: dict | None = None
) -> dict:
    """組出 events.json 的頂層結構。"""
    out = {
        "source": source,
        "fps": fps,
        "n_events": len(events),
        "events": [e.to_dict() for e in events],
    }
    if extra:
        out.update(extra)
    return out


def write_events_json(
    path: str | Path, events: list[FallEvent], source: str, fps: float, extra: dict | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(events_to_json_dict(events, source, fps, extra), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
