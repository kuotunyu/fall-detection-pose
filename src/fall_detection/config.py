"""config.yaml 的載入與驗證。

所有閾值的單位與選擇理由以 config.yaml 內的註解為準;本模組只負責型別與
一致性檢查。非法組合(例如遲滯出口閾值不低於進入閾值)直接 fail-fast,
不做隱式修正——評估數字的可信度建立在設定檔與程式行為完全一致之上。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """推論模型與追蹤器參數(僅 extract 階段使用)。"""

    name: str = "yolo26n-pose.pt"
    conf: float = Field(0.25, gt=0.0, lt=1.0)
    iou: float = Field(0.5, gt=0.0, lt=1.0)
    tracker: str = "bytetrack.yaml"
    kpt_conf_min: float = Field(0.35, ge=0.0, lt=1.0)


class RulesConfig(BaseModel):
    """規則引擎閾值。長度單位=軀幹長 L̃、時間單位=秒、角度單位=度。"""

    smooth_s: float = Field(gt=0.0)
    velocity_window_s: float = Field(gt=0.0)
    v_fall_enter: float = Field(gt=0.0)
    omega_enter: float = Field(gt=0.0)
    theta_lying_enter: float = Field(gt=0.0, lt=90.0)
    theta_upright_exit: float = Field(gt=0.0, lt=90.0)
    r_lying: float = Field(gt=0.0)
    h_hip_lying: float = Field(gt=0.0)
    h_hip_upright_exit: float = Field(gt=0.0)
    posture_votes_required: int = Field(ge=1, le=3)
    window_confirm_s: float = Field(gt=0.0)
    vote_ratio: float = Field(gt=0.0, le=1.0)
    t_falling_timeout_s: float = Field(gt=0.0)
    t_confirm_fallen_s: float = Field(ge=0.0)
    t_recover_s: float = Field(gt=0.0)
    max_kpt_gap_s: float = Field(ge=0.0)
    track_lost_timeout_s: float = Field(gt=0.0)
    track_stitch_iou: float = Field(gt=0.0, lt=1.0)
    track_stitch_window_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "RulesConfig":
        if self.theta_upright_exit >= self.theta_lying_enter:
            raise ValueError(
                "theta_upright_exit 必須小於 theta_lying_enter(遲滯出口需低於進入閾值)"
            )
        if self.h_hip_upright_exit <= self.h_hip_lying:
            raise ValueError(
                "h_hip_upright_exit 必須大於 h_hip_lying(遲滯出口需高於進入閾值)"
            )
        if self.max_kpt_gap_s > self.track_lost_timeout_s:
            raise ValueError(
                "max_kpt_gap_s 不可大於 track_lost_timeout_s(hold-last 不能比 track 終結還久)"
            )
        return self


class EventsConfig(BaseModel):
    """事件後處理參數。"""

    min_event_duration_s: float = Field(ge=0.0)
    merge_gap_s: float = Field(ge=0.0)


class Config(BaseModel):
    model: ModelConfig
    rules: RulesConfig
    events: EventsConfig


def load_config(path: str | Path) -> Config:
    """讀取並驗證 YAML 設定檔;任何缺欄位或非法值都拋出例外。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
