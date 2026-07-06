"""狀態機的直接單元測試(手工 TickInput 序列,不經特徵管線)。"""

from fall_detection.rules.state_machine import FallStateMachine, State, TickInput

DT = 1.0 / 30.0

UPRIGHT_KW = dict(theta_deg=5.0, bbox_aspect=0.4, h_hip=1.8, v_norm=0.0, omega=0.0)
LYING_KW = dict(theta_deg=85.0, bbox_aspect=2.5, h_hip=0.1, v_norm=0.2, omega=0.0)
SIT_KW = dict(theta_deg=20.0, bbox_aspect=0.6, h_hip=0.9, v_norm=0.0, omega=0.0)


def _feed(fsm, t0, dur, frame0, **kw):
    """以 30fps 連續餵同一組特徵 dur 秒;回傳 (下一時間, 下一幀號)。"""
    n = round(dur / DT)
    for i in range(n):
        t = t0 + i * DT
        fsm.tick(TickInput(t_s=t, frame_idx=frame0 + i, **kw))
    return t0 + n * DT, frame0 + n


def _fall_to_alarm(cfg):
    """站立 → 觸發 → 躺地至 ALARM,回傳 (fsm, t, frame)。"""
    fsm = FallStateMachine(cfg.rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=30.0, bbox_aspect=0.8, h_hip=1.2, v_norm=3.0, omega=120.0))
    assert fsm.state is State.FALLING
    t, f = t + DT, f + 1
    t, f = _feed(fsm, t, 2.0, f, **LYING_KW)
    assert fsm.state is State.ALARM
    return fsm, t, f


def test_full_fall_path_and_recovery_closes_event(cfg):
    fsm, t, f = _fall_to_alarm(cfg)
    t, f = _feed(fsm, t, cfg.rules.t_recover_s + 0.2, f, **UPRIGHT_KW)
    assert fsm.state is State.UPRIGHT
    events = fsm.finalize()
    assert len(events) == 1
    ev = events[0]
    assert ev.track_ids == [1]
    assert "v>v_fall_enter" in ev.rules_fired
    assert "lying_persisted" in ev.rules_fired
    assert ev.start_time_s <= 1.1  # 事件起點 = 進 FALLING 的時刻
    assert ev.peak_features["max_v_torso_per_s"] >= 3.0


def test_track_end_closes_open_alarm(cfg):
    fsm, _, _ = _fall_to_alarm(cfg)
    events = fsm.finalize()
    assert len(events) == 1  # 影片在 ALARM 中結束:事件以最後一幀收尾


def test_falling_timeout_rejects_fast_sit(cfg):
    fsm = FallStateMachine(cfg.rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=15.0, bbox_aspect=0.5, h_hip=1.0, v_norm=2.5, omega=30.0))
    assert fsm.state is State.FALLING
    t, f = _feed(fsm, t + DT, cfg.rules.t_falling_timeout_s + 0.3, f + 1, **SIT_KW)
    assert fsm.state is State.UPRIGHT
    assert fsm.finalize() == []


def test_fallen_recovery_before_confirm_yields_no_event(cfg):
    """進了 FALLEN 但在 t_confirm_fallen_s 前回正:不出事件(去抖動的核心)。"""
    rules = cfg.rules.model_copy(update={"t_confirm_fallen_s": 2.0})
    fsm = FallStateMachine(rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=30.0, bbox_aspect=0.8, h_hip=1.0, v_norm=3.0, omega=100.0))
    t, f = _feed(fsm, t + DT, 0.6, f + 1, **LYING_KW)
    assert fsm.state is State.FALLEN
    t, f = _feed(fsm, t, 1.0, f, **UPRIGHT_KW)
    assert fsm.state is State.UPRIGHT
    assert fsm.finalize() == []


def test_hysteresis_no_chatter_around_lying_threshold(cfg):
    """θ 在躺姿進入閾值(60°)附近震盪:遲滯出口(40°)未觸及 → 狀態不抖、事件不碎。"""
    fsm, t, f = _fall_to_alarm(cfg)
    states = []
    n = round(3.0 / DT)
    for i in range(n):
        theta = 62.0 if i % 2 == 0 else 58.0
        h_hip = 0.55 if i % 2 == 0 else 0.45
        fsm.tick(TickInput(t_s=t + i * DT, frame_idx=f + i, theta_deg=theta,
                           bbox_aspect=1.2, h_hip=h_hip, v_norm=0.0, omega=0.0))
        states.append(fsm.state)
    assert all(s is State.ALARM for s in states)  # 震盪期間狀態穩定
    events = fsm.finalize()
    assert len(events) == 1  # 事件沒有被切碎


def test_none_inputs_do_not_trigger_or_crash(cfg):
    """v/omega/h_hip 為 None(歷史不足、踝不可見):不觸發任何條件、不拋例外。"""
    fsm = FallStateMachine(cfg.rules, track_id=1)
    for i in range(60):
        fsm.tick(TickInput(t_s=i * DT, frame_idx=i, theta_deg=5.0,
                           bbox_aspect=0.4, h_hip=None, v_norm=None, omega=None))
    assert fsm.state is State.UPRIGHT
    assert fsm.finalize() == []


def test_adopt_records_track_chain(cfg):
    fsm, t, f = _fall_to_alarm(cfg)
    fsm.adopt(7)
    events = fsm.finalize()
    assert events[0].track_ids == [1, 7]
