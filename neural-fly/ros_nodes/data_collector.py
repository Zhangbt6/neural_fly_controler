#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural-Fly 数据采集节点。

订阅 MuJoCo 仿真器 (或真实飞行) 发布的里程计与电机指令，按 Neural-Fly/DAIML
所需字段实时计算气动残差力 `fa`，并保存成官方训练脚本可直接使用的 csv。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

# 允许直接导入 neural-fly/* 模块
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))  # noqa: E402

import utils  # type: ignore
from feature_utils import (AlphaFilter, compute_residual_force,
                           compute_total_thrust, quat_to_rot_matrix)


@dataclass
class SampleBuffer:
    """暂存一段飞行数据并最终写入 csv。"""

    t: List[float] = field(default_factory=list)
    p: List[List[float]] = field(default_factory=list)
    v: List[List[float]] = field(default_factory=list)
    q: List[List[float]] = field(default_factory=list)
    pwm: List[List[float]] = field(default_factory=list)
    fa: List[List[float]] = field(default_factory=list)
    T_sp: List[float] = field(default_factory=list)
    hover_throttle: List[float] = field(default_factory=list)

    def clear(self):
        for attr in vars(self).values():
            attr.clear()

    def size(self) -> int:
        return len(self.t)


def parse_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--log-directory', dest='log_directory', type=str)
    parser.add_argument('--vehicle', type=str)
    parser.add_argument('--trajectory', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--condition', type=str)
    parser.add_argument('--mass', type=float)
    parser.add_argument('--gravity', type=float)
    parser.add_argument('--motor-force-scale', dest='motor_force_scale', type=float)
    parser.add_argument('--hover-throttle', dest='hover_throttle', type=float)
    parser.add_argument('--acc-filter-alpha', dest='acc_filter_alpha', type=float)

    # 允许保留其它 ROS remap 参数
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


class NeuralFlyDataCollector:
    def __init__(self):
        self.cli_args = parse_cli_args()
        rospy.init_node('neural_fly_data_collector', anonymous=False)

        # 参数配置
        self.odom_topic = rospy.get_param('~odom_topic', '/quad/odometry')
        self.motor_topic = rospy.get_param('~motor_cmd_topic', '/quad/motor_cmd')

        self.mass = self._param_with_cli('mass', 2.458)
        self.gravity = self._param_with_cli('gravity', 9.8066)
        self.motor_force_scale = self._param_with_cli('motor_force_scale', 12.5)
        self.hover_throttle_default = self._param_with_cli('hover_throttle', 0.5)
        self.acc_alpha = self._param_with_cli('acc_filter_alpha', 0.35)

        log_dir_default = str(PACKAGE_ROOT / 'data' / 'sim_log')
        log_dir_value = self.cli_args.log_directory if self.cli_args.log_directory else rospy.get_param('~log_directory', log_dir_default)
        log_dir_param = log_dir_value
        self.log_directory = Path(log_dir_param).expanduser()
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # 关键 metadata（用于文件命名）
        self.vehicle = self._param_with_cli('vehicle', 'sim')
        self.trajectory = self._param_with_cli('trajectory', 'hover')
        self.method = self._param_with_cli('method', 'NF-sim')
        self.condition = self._param_with_cli('condition', 'nowind')

        self.buffer = SampleBuffer()
        self.log_active = False

        self.prev_time: Optional[float] = None
        self.prev_velocity: Optional[np.ndarray] = None
        self.acc_filter = AlphaFilter(self.acc_alpha)

        self.last_motor_cmd: Optional[np.ndarray] = None

        # 订阅话题
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=50)
        self.motor_sub = rospy.Subscriber(self.motor_topic, Float64MultiArray, self._motor_callback, queue_size=50)

        # 服务：开始/停止记录
        self.start_srv = rospy.Service('~start', SetBool, self._start_logging)
        self.stop_srv = rospy.Service('~stop_and_save', Trigger, self._stop_and_save)

        rospy.loginfo('[neural_fly_data_collector] 节点启动，等待 start/stop 指令。')

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def _motor_callback(self, msg: Float64MultiArray):
        data = np.array(msg.data, dtype=float)
        if data.shape[0] != 4:
            rospy.logwarn_throttle(5.0, '[data_collector] motor_cmd 长度非 4，忽略。')
            return
        self.last_motor_cmd = data

    def _odom_callback(self, msg: Odometry):
        if not self.log_active:
            return
        if self.last_motor_cmd is None:
            rospy.logwarn_throttle(5.0, '[data_collector] 未收到 motor_cmd，无法计算残差力。')
            return

        t = msg.header.stamp.to_sec()
        pos = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z], dtype=float)
        vel_world = np.array([msg.twist.twist.linear.x,
                              msg.twist.twist.linear.y,
                              msg.twist.twist.linear.z], dtype=float)
        quat_wxyz = np.array([msg.pose.pose.orientation.w,
                               msg.pose.pose.orientation.x,
                               msg.pose.pose.orientation.y,
                               msg.pose.pose.orientation.z], dtype=float)

        rot_bw = quat_to_rot_matrix(quat_wxyz)

        if self.prev_time is None:
            self.prev_time = t
            self.prev_velocity = vel_world
            self.acc_filter.reset(np.zeros(3))
            return

        dt = t - self.prev_time
        if dt <= 1e-6:
            return

        raw_acc = (vel_world - self.prev_velocity) / dt
        acc_world = self.acc_filter.update(raw_acc)

        total_thrust = compute_total_thrust(self.last_motor_cmd, self.motor_force_scale)
        residual_force = compute_residual_force(mass=self.mass,
                                                acc_world=acc_world,
                                                rot_bw=rot_bw,
                                                total_thrust=total_thrust,
                                                gravity=self.gravity)

        self.buffer.t.append(t)
        self.buffer.p.append(pos.tolist())
        self.buffer.v.append(vel_world.tolist())
        # 与官方数据保持一致：四元数写成列表 [w, x, y, z]
        self.buffer.q.append(quat_wxyz.tolist())
        self.buffer.pwm.append(self.last_motor_cmd.tolist())
        self.buffer.fa.append(residual_force.tolist())
        self.buffer.T_sp.append(total_thrust)
        self.buffer.hover_throttle.append(self.hover_throttle_default)

        self.prev_time = t
        self.prev_velocity = vel_world

    # ------------------------------------------------------------------
    # 服务
    # ------------------------------------------------------------------
    def _start_logging(self, req: SetBool):
        if req.data:
            self._reset_buffer()
            self.log_active = True
            rospy.loginfo('[data_collector] 开始记录。')
            return SetBoolResponse(success=True, message='logging started')
        else:
            self.log_active = False
            rospy.loginfo('[data_collector] 已暂停记录（未保存）。')
            return SetBoolResponse(success=True, message='logging paused')

    def _stop_and_save(self, _req: Trigger):
        self.log_active = False
        if self.buffer.size() < 10:
            self._reset_buffer()
            return TriggerResponse(success=False, message='buffer too small, nothing saved')

        record = {
            't': np.array(self.buffer.t),
            'p': np.array(self.buffer.p),
            'v': np.array(self.buffer.v),
            'q': np.array(self.buffer.q),
            'pwm': np.array(self.buffer.pwm),
            'fa': np.array(self.buffer.fa),
            'hover_throttle': np.array(self.buffer.hover_throttle),
            'T_sp': np.array(self.buffer.T_sp)[:, None],
            'vehicle': self.vehicle,
            'trajectory': self.trajectory,
            'method': self.method,
            'condition': self.condition,
        }

        filename = f"{self.vehicle}_{self.trajectory}_{self.method}_{self.condition}"
        utils.save_data([record], str(self.log_directory),
                        fields=['t', 'p', 'v', 'q', 'pwm', 'fa', 'hover_throttle', 'T_sp'])

        saved_path = os.path.join(str(self.log_directory), filename + '.csv')
        rospy.loginfo('[data_collector] 已保存数据到 %s', saved_path)
        self._reset_buffer()
        return TriggerResponse(success=True, message=saved_path)

    def _reset_buffer(self):
        self.buffer.clear()
        self.prev_time = None
        self.prev_velocity = None
        self.acc_filter.reset()

    def _param_with_cli(self, name: str, default):
        cli_value = getattr(self.cli_args, name, None)
        if cli_value is not None:
            return cli_value
        return rospy.get_param(f'~{name}', default)


def main():
    collector = NeuralFlyDataCollector()
    rospy.spin()


if __name__ == '__main__':
    main()
