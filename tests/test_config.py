"""設定檔載入與一致性驗證的測試。"""

import pytest
import yaml
from pydantic import ValidationError

from fall_detection.config import Config, load_config

from conftest import REPO_ROOT


def test_repo_config_loads(cfg):
    assert cfg.model.name.endswith("-pose.pt")
    assert 0 < cfg.model.kpt_conf_min < 1
    assert cfg.rules.theta_upright_exit < cfg.rules.theta_lying_enter


def test_hysteresis_violation_rejected():
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["rules"]["theta_upright_exit"] = raw["rules"]["theta_lying_enter"] + 5
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_hip_hysteresis_violation_rejected():
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["rules"]["h_hip_upright_exit"] = raw["rules"]["h_hip_lying"] - 0.1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_missing_field_rejected():
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    del raw["rules"]["v_fall_enter"]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_events_postprocess_merge_and_min_duration(cfg):
    from fall_detection.events.schema import FallEvent, postprocess_events

    a = FallEvent([1], 60, 90, 2.0, 3.0, {"max_v_torso_per_s": 2.0}, ["v>v_fall_enter"])
    b = FallEvent([1, 7], 105, 150, 3.5, 5.0, {"max_v_torso_per_s": 3.0}, ["lying_persisted"])
    c = FallEvent([2], 200, 202, 6.6, 6.7, {}, [])  # 0.1s 抖動殘渣
    out = postprocess_events([c, b, a], cfg.events)  # 順序故意打亂
    assert len(out) == 1  # a+b 合併(同 track 鏈、gap 0.5 < 1.0);c 被濾掉
    ev = out[0]
    assert ev.track_ids == [1, 7]
    assert ev.start_time_s == 2.0 and ev.end_time_s == 5.0
    assert ev.peak_features["max_v_torso_per_s"] == 3.0


def test_no_merge_across_different_tracks(cfg):
    from fall_detection.events.schema import FallEvent, postprocess_events

    a = FallEvent([1], 60, 90, 2.0, 3.0, {}, [])
    b = FallEvent([2], 100, 150, 3.4, 5.0, {}, [])
    out = postprocess_events([a, b], cfg.events)
    assert len(out) == 2  # track 鏈無交集:不合併


def test_merge_not_blocked_by_interleaved_other_chain(cfg):
    """多人場景:別人的事件插在同鏈兩片段之間,不得擋住合併(review 抓到的 bug)。"""
    from fall_detection.events.schema import FallEvent, postprocess_events

    a1 = FallEvent([1], 60, 90, 2.0, 3.0, {}, [])
    b = FallEvent([2], 96, 114, 3.2, 3.8, {}, [])   # 依 start 排序會插在 a1 與 a2 之間
    a2 = FallEvent([1], 105, 150, 3.5, 5.0, {}, [])  # 與 a1 間隔 0.5s < merge_gap_s
    out = postprocess_events([a1, b, a2], cfg.events)
    assert len(out) == 2
    chain1 = next(e for e in out if 1 in e.track_ids)
    assert (chain1.start_time_s, chain1.end_time_s) == (2.0, 5.0)  # a1+a2 有合併
