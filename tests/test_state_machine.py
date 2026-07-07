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


def test_finalize_while_fallen_still_closes_event(cfg):
    """FALLEN(躺姿已投票確認)但尚未撐滿 t_confirm_fallen_s 就整個 track 消失:
    仍應收尾為一次事件,而非靜默丟棄(躺地後 pose 模型偵測斷訊是已知風險,
    寧可觸發不可漏判)。"""
    rules = cfg.rules.model_copy(update={"t_confirm_fallen_s": 5.0})
    fsm = FallStateMachine(rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=30.0, bbox_aspect=0.8, h_hip=1.2, v_norm=3.0, omega=120.0))
    t, f = _feed(fsm, t + DT, 0.6, f + 1, **LYING_KW)
    assert fsm.state is State.FALLEN
    events = fsm.finalize()
    assert len(events) == 1
    assert "track_lost_while_fallen" in events[0].rules_fired
    assert "posture_vote_confirmed" in events[0].rules_fired
    assert "lying_persisted" not in events[0].rules_fired  # 沒撐到 ALARM,不該有這個 tag


def test_finalize_while_falling_with_lying_posture_closes_event(cfg):
    """FALLING 中 track 消失,躺姿投票視窗還沒累積夠樣本確認,但最後一次
    平滑觀測已符合躺姿 m-of-n:仍應收尾為一次事件——這正是 URFD test split
    fall-24 踩到的真實案例(訊號都到位,只是資料在視窗跑完前就斷了)。"""
    fsm = FallStateMachine(cfg.rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=30.0, bbox_aspect=0.8, h_hip=1.2, v_norm=3.0, omega=120.0))
    assert fsm.state is State.FALLING
    t, f = t + DT, f + 1
    # 只餵 1 幀躺姿(遠不足以撐滿 window_confirm_s 的時窗投票),track 就此消失
    fsm.tick(TickInput(t_s=t, frame_idx=f, **LYING_KW))
    events = fsm.finalize()
    assert len(events) == 1
    assert "track_lost_while_falling_with_lying_posture" in events[0].rules_fired
    assert "posture_vote_confirmed" not in events[0].rules_fired  # 投票視窗真的沒跑完


def test_finalize_while_falling_without_lying_posture_no_event(cfg):
    """負向對照:FALLING 中 track 消失,但最後觀測並不符合躺姿(如坐姿):
    不該生出事件——避免這個新規則過度寬鬆,只認最後一次真的像躺姿的情況。"""
    fsm = FallStateMachine(cfg.rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=30.0, bbox_aspect=0.8, h_hip=1.2, v_norm=3.0, omega=120.0))
    assert fsm.state is State.FALLING
    t, f = t + DT, f + 1
    fsm.tick(TickInput(t_s=t, frame_idx=f, **SIT_KW))  # 坐姿:不符合躺姿 m-of-n
    assert fsm.finalize() == []


def test_bridge_gap_prevents_false_falling_timeout(cfg):
    """縫合空窗若不橋接,消失期間的時間差會被誤算成「觀察了這麼久還沒確認」,
    導致 falling-timeout 誤回退——這正是 URFD fall-01 煙測踩到的真實案例
    (track 在觸發 FALLING 後消失超過 1s,縫合回來時真實經過時間已超過
    t_falling_timeout_s)。"""
    fsm = FallStateMachine(cfg.rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=30.0, bbox_aspect=0.8, h_hip=1.2, v_norm=3.0, omega=120.0))
    assert fsm.state is State.FALLING
    falling_since_before = fsm._falling_since
    # 消失空窗:真實經過時間 > t_falling_timeout_s,不橋接就會誤觸發回退
    gap = cfg.rules.t_falling_timeout_s + 0.5
    fsm.adopt(9)
    fsm.bridge_gap(gap)
    assert fsm._falling_since == falling_since_before + gap
    t2 = t + DT + gap
    fsm.tick(TickInput(t_s=t2, frame_idx=f + 1, theta_deg=85.0, bbox_aspect=2.5, h_hip=0.1, v_norm=0.2, omega=0.0))
    assert fsm.state is State.FALLING  # 沒被誤回退成 UPRIGHT
    assert fsm.track_ids == [1, 9]


def test_bridge_gap_does_not_pollute_vote_window_with_stale_samples(cfg):
    """bridge_gap 位移 _falling_since 等單一時間戳,但刻意不位移 _vote_win:
    否則空窗前的舊「未躺」樣本會被誤認成剛觀察到,拖累縫合後的躺姿投票比例,
    延誤本該在 window_confirm_s 內就能確認的 FALLEN 判定。"""
    fsm = FallStateMachine(cfg.rules, track_id=1)
    t, f = _feed(fsm, 0.0, 1.0, 0, **UPRIGHT_KW)
    fsm.tick(TickInput(t_s=t, frame_idx=f, theta_deg=30.0, bbox_aspect=0.8, h_hip=1.2, v_norm=3.0, omega=120.0))
    # 短暫停留 FALLING,累積幾幀「尚未躺平」的 False 投票樣本
    t, f = _feed(fsm, t + DT, 0.3, f + 1, theta_deg=40.0, bbox_aspect=0.7, h_hip=1.0, v_norm=2.0, omega=0.0)
    assert fsm.state is State.FALLING
    fsm.adopt(9)
    fsm.bridge_gap(1.5)
    t2 = t + 1.5
    # 縫合後立刻是清楚的躺姿,只餵 window_confirm_s 多一點:若舊樣本沒被正確
    # 排除,比例會被拖低而確認不了
    t2, f = _feed(fsm, t2, cfg.rules.window_confirm_s + 0.2, f, **LYING_KW)
    assert fsm.state is State.FALLEN


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
