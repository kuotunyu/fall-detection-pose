"""每個 track 一台跌倒狀態機。

純邏輯:輸入為已平滑、已正規化的特徵(見 ``engine``),輸出狀態與事件。

    UPRIGHT ─(v>v_fall_enter 或 ω>omega_enter)→ FALLING
    FALLING ─(躺姿 m-of-n 投票在時窗內達 vote_ratio)→ FALLEN
    FALLING ─(t_falling_timeout_s 內未確認)→ UPRIGHT(回退,不出事件:擋快速坐下/蹲下)
    FALLEN  ─(躺姿連續 t_confirm_fallen_s)→ ALARM(此刻才「確認」一次跌倒)
    FALLEN/ALARM ─(回正持續 t_recover_s;遲滯出口閾值)→ UPRIGHT
                  (ALARM 時關閉事件;FALLEN 未達確認,不出事件)

事件的起點是「進入 FALLING 的幀」(跌倒開始),而非 ALARM 的幀——
告警需要去抖動延遲,但事件時間軸要對齊真實跌倒,評估才公平。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from ..config import RulesConfig
from ..events.schema import FallEvent


class State(str, Enum):
    UPRIGHT = "UPRIGHT"
    FALLING = "FALLING"
    FALLEN = "FALLEN"
    ALARM = "ALARM"


@dataclass
class TickInput:
    """一次狀態機更新的輸入(皆為平滑/正規化後的值)。

    ``h_hip``/``v_norm``/``omega`` 為 None 表示該值本幀不可得
    (踝不可見、歷史不足):None 不觸發任何條件,也不投票。
    """

    t_s: float
    frame_idx: int
    theta_deg: float
    bbox_aspect: float
    h_hip: float | None
    v_norm: float | None
    omega: float | None


@dataclass
class _FallContext:
    """從 FALLING 進入開始累積的事件上下文;回退時整包丟棄。"""

    start_frame: int
    start_t: float
    rules_fired: set
    max_v: float = float("-inf")
    max_theta: float = float("-inf")


class FallStateMachine:
    def __init__(self, cfg: RulesConfig, track_id: int):
        self.cfg = cfg
        self.track_ids: list[int] = [int(track_id)]
        self.state: State = State.UPRIGHT
        self.completed: list[FallEvent] = []

        self._vote_win: deque[tuple[float, bool]] = deque()
        self._ctx: _FallContext | None = None
        self._falling_since: float | None = None
        self._lying_since: float | None = None
        self._not_lying_since: float | None = None
        self._recover_since: float | None = None
        self._alarm_open: bool = False
        self._last_t: float | None = None
        self._last_frame: int | None = None

    # ---------- 對外 ----------

    def adopt(self, track_id: int) -> None:
        """track 縫合:新 id 繼承本狀態機(事件會記錄整條 id 鏈)。"""
        tid = int(track_id)
        if tid not in self.track_ids:
            self.track_ids.append(tid)

    def tick(self, x: TickInput) -> None:
        cfg = self.cfg
        self._last_t, self._last_frame = x.t_s, x.frame_idx

        lying_now = self._posture_lying(x)
        self._push_vote(x.t_s, lying_now)
        self._update_recover(x, lying_now)
        if self._ctx is not None:
            if x.v_norm is not None:
                self._ctx.max_v = max(self._ctx.max_v, x.v_norm)
            self._ctx.max_theta = max(self._ctx.max_theta, x.theta_deg)

        if self.state is State.UPRIGHT:
            trig = []
            if x.v_norm is not None and x.v_norm > cfg.v_fall_enter:
                trig.append("v>v_fall_enter")
            if x.omega is not None and x.omega > cfg.omega_enter:
                trig.append("omega>omega_enter")
            if trig:
                self.state = State.FALLING
                self._falling_since = x.t_s
                self._ctx = _FallContext(
                    start_frame=x.frame_idx,
                    start_t=x.t_s,
                    rules_fired=set(trig),
                    max_v=x.v_norm if x.v_norm is not None else float("-inf"),
                    max_theta=x.theta_deg,
                )

        elif self.state is State.FALLING:
            if self._vote_confirmed(x.t_s):
                self.state = State.FALLEN
                self._ctx.rules_fired.add("posture_vote_confirmed")
                self._lying_since = x.t_s
                self._not_lying_since = None
            elif x.t_s - self._falling_since > cfg.t_falling_timeout_s:
                # 未確認躺姿:視為快速坐下/蹲下,回退且不出事件
                self._rollback()

        elif self.state is State.FALLEN:
            if lying_now:
                self._lying_since = self._lying_since if self._lying_since is not None else x.t_s
                self._not_lying_since = None
            else:
                self._lying_since = None
                self._not_lying_since = (
                    self._not_lying_since if self._not_lying_since is not None else x.t_s
                )
            if (
                self._lying_since is not None
                and x.t_s - self._lying_since >= cfg.t_confirm_fallen_s
            ):
                self.state = State.ALARM
                self._alarm_open = True
                self._ctx.rules_fired.add("lying_persisted")
            elif self._recover_sustained(x.t_s):
                self._rollback()  # 未達 ALARM 即回正:不出事件
            elif (
                self._not_lying_since is not None
                and x.t_s - self._not_lying_since > cfg.t_falling_timeout_s
            ):
                self._rollback()  # 軟重置:半躺不躺(如跌成坐姿)久滯,不告警也不卡死

        elif self.state is State.ALARM:
            if self._recover_sustained(x.t_s):
                self._close_event(x.frame_idx, x.t_s)
                self._rollback()

    def finalize(self) -> list[FallEvent]:
        """track 結束(消失逾時或影片結尾):關閉進行中的 ALARM 事件並回傳全部事件。"""
        if self._alarm_open and self._last_frame is not None:
            self._close_event(self._last_frame, self._last_t)
        self._ctx = None
        self._alarm_open = False
        out, self.completed = self.completed, []
        return out

    # ---------- 內部 ----------

    def _posture_lying(self, x: TickInput) -> bool:
        """躺姿 m-of-n 投票;踝不可見時 h_hip 不投票(不硬猜)。"""
        cfg = self.cfg
        votes = [
            x.theta_deg > cfg.theta_lying_enter,
            x.bbox_aspect > cfg.r_lying,
        ]
        if x.h_hip is not None:
            votes.append(x.h_hip < cfg.h_hip_lying)
        return sum(votes) >= cfg.posture_votes_required

    def _push_vote(self, t: float, lying: bool) -> None:
        self._vote_win.append((t, lying))
        cutoff = t - self.cfg.window_confirm_s
        while self._vote_win and self._vote_win[0][0] < cutoff - 1e-9:
            self._vote_win.popleft()

    def _vote_confirmed(self, t: float) -> bool:
        if len(self._vote_win) < 2:
            return False
        span = t - self._vote_win[0][0]
        if span < 0.5 * self.cfg.window_confirm_s:
            return False  # 窗內樣本太少,單幀雜訊也能過票——先不確認
        ratio = sum(1 for _, ly in self._vote_win if ly) / len(self._vote_win)
        return ratio >= self.cfg.vote_ratio

    def _update_recover(self, x: TickInput, lying_now: bool) -> None:
        cfg = self.cfg
        recovered_now = (
            not lying_now
            and x.theta_deg < cfg.theta_upright_exit
            and (x.h_hip is None or x.h_hip > cfg.h_hip_upright_exit)
        )
        if recovered_now:
            if self._recover_since is None:
                self._recover_since = x.t_s
        else:
            self._recover_since = None

    def _recover_sustained(self, t: float) -> bool:
        return (
            self._recover_since is not None
            and t - self._recover_since >= self.cfg.t_recover_s
        )

    def _close_event(self, end_frame: int, end_t: float) -> None:
        ctx = self._ctx
        peaks = {}
        if ctx.max_v != float("-inf"):
            peaks["max_v_torso_per_s"] = round(ctx.max_v, 3)
        if ctx.max_theta != float("-inf"):
            peaks["max_theta_deg"] = round(ctx.max_theta, 1)
        self.completed.append(
            FallEvent(
                track_ids=sorted(self.track_ids),
                start_frame=ctx.start_frame,
                end_frame=int(end_frame),
                start_time_s=round(ctx.start_t, 3),
                end_time_s=round(float(end_t), 3),
                peak_features=peaks,
                rules_fired=sorted(ctx.rules_fired),
            )
        )
        self._alarm_open = False

    def _rollback(self) -> None:
        """回到 UPRIGHT 並清空事件上下文(已關閉的事件保留在 completed)。"""
        self.state = State.UPRIGHT
        self._ctx = None
        self._falling_since = None
        self._lying_since = None
        self._not_lying_since = None
        self._alarm_open = False
