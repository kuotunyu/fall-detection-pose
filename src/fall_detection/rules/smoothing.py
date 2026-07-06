"""時間域平滑與固定時窗差分的小工具(無狀態機邏輯,只做數值處理)。

差分一律以「時間」為窗(而非幀數),確保換 fps 不需重調閾值。
"""

from __future__ import annotations

from collections import deque


class RollingMedian:
    """固定長度滑動中位數;窗未滿時回傳現有樣本的中位數。

    選中位數而非平均:單幀關鍵點跳動(離群值)會拉壞平均,
    中位數在 5 幀窗內即可壓掉單幀 outlier 而不引入明顯延遲。
    """

    def __init__(self, window: int):
        self._buf: deque[float] = deque(maxlen=max(1, int(window)))

    def push(self, x: float) -> float:
        self._buf.append(float(x))
        s = sorted(self._buf)
        n = len(s)
        if n % 2:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    def __len__(self) -> int:
        return len(self._buf)


class TimedBuffer:
    """(t, value) 緩衝:支援取「t−Δ 附近」的樣本做固定時窗差分。"""

    def __init__(self, horizon_s: float):
        self.horizon_s = float(horizon_s)
        self._buf: deque[tuple[float, float]] = deque()

    def push(self, t: float, v: float) -> None:
        self._buf.append((float(t), float(v)))
        while self._buf and self._buf[0][0] < t - self.horizon_s - 1e-9:
            self._buf.popleft()

    def sample_at_or_before(self, t_query: float) -> tuple[float, float] | None:
        """時間戳 ≤ t_query 的最新樣本;若全部樣本都比 t_query 新,退回最舊樣本
        (讓差分在歷史稍短時仍可用,由呼叫端以最短時距把關)。"""
        if not self._buf:
            return None
        best: tuple[float, float] | None = None
        for t, v in self._buf:
            if t <= t_query + 1e-9:
                best = (t, v)
            else:
                break
        return best if best is not None else self._buf[0]

    def rate(self, t_now: float, v_now: float, delta_s: float) -> float | None:
        """(v_now − v(t_now−Δ)) / 實際時距;歷史不足(時距 < Δ/2)回 None。"""
        got = self.sample_at_or_before(t_now - delta_s)
        if got is None:
            return None
        t0, v0 = got
        span = t_now - t0
        if span <= 0 or span < 0.5 * delta_s:
            return None
        return (v_now - v0) / span
