#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工具函数：统一特征/扰动计算逻辑，便于数据采集与在线学习节点复用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from tf.transformations import quaternion_matrix


def quat_to_rot_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """将四元数 (w, x, y, z) 转换为 3x3 旋转矩阵（body -> world）。"""
    if q_wxyz.shape[-1] != 4:
        raise ValueError("Quaternion must have length 4 (w, x, y, z)")
    q = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # (x, y, z, w)
    rot = quaternion_matrix(q)[:3, :3]
    return rot


def world_to_body_velocity(vel_world: np.ndarray, rot_bw: np.ndarray) -> np.ndarray:
    """把世界系线速度转换到机体系: v_b = R^T v_w。"""
    return rot_bw.T @ vel_world


def compute_total_thrust(throttle: np.ndarray, motor_force_scale: float) -> float:
    """将 0-1 归一化油门换算为总推力 (N)。"""
    return float(np.sum(throttle) * motor_force_scale)


def compute_residual_force(*,
                           mass: float,
                           acc_world: np.ndarray,
                           rot_bw: np.ndarray,
                           total_thrust: float,
                           gravity: float) -> np.ndarray:
    """根据 m·a - m·g - R·[0,0,T]^T 计算气动残差力。"""
    g_vec = np.array([0.0, 0.0, gravity])
    thrust_world = rot_bw @ np.array([0.0, 0.0, total_thrust])
    return mass * acc_world - mass * g_vec - thrust_world


@dataclass
class AlphaFilter:
    """一阶低通滤波器，常用于速度差分求加速度。"""

    alpha: float
    init_value: Optional[np.ndarray] = None

    def __post_init__(self):
        self._state: Optional[np.ndarray] = None
        if self.init_value is not None:
            self._state = np.array(self.init_value, dtype=float)

    def reset(self, value: Optional[np.ndarray] = None):
        self._state = None if value is None else np.array(value, dtype=float)

    def update(self, new_value: np.ndarray) -> np.ndarray:
        new_value = np.array(new_value, dtype=float)
        if self._state is None:
            self._state = new_value.copy()
        else:
            self._state = self.alpha * new_value + (1.0 - self.alpha) * self._state
        return self._state.copy()

