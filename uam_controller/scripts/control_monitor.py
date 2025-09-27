#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制监控节点：订阅控制器输出的调试话题，并通过 Matplotlib 实时绘制关键数据。
可直接配合 `uam_controller` 包发布的 /uam_controller/debug/* 话题使用。
"""
import collections
import threading
from typing import Deque, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64


class TimeSeriesBuffer:
    """简单的环形缓存，用于存储时间序列数据。"""

    def __init__(self, max_len: int = 600):
        self.times: Deque[float] = collections.deque(maxlen=max_len)
        self.values: Deque[List[float]] = collections.deque(maxlen=max_len)

    def append(self, stamp: rospy.Time, data: List[float]):
        self.times.append(stamp.to_sec())
        self.values.append(data)

    def as_arrays(self):
        if not self.times:
            return np.array([]), np.empty((0, len(self.values[0]) if self.values else 0))
        t0 = self.times[0]
        t = np.array(self.times) - t0
        arr = np.vstack(self.values)
        return t, arr


class ControlMonitor:
    def __init__(self):
        rospy.init_node("control_monitor", anonymous=True)

        max_len = rospy.get_param("~buffer_size", 600)
        self.update_rate = rospy.get_param("~update_rate", 10.0)

        # 配置中文字体，若系统缺少对应字体将自动回落
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

        self.buffers: Dict[str, TimeSeriesBuffer] = {
            "position": TimeSeriesBuffer(max_len=max_len),
            "velocity": TimeSeriesBuffer(max_len=max_len),
            "position_error": TimeSeriesBuffer(max_len=max_len),
            "accel_cmd": TimeSeriesBuffer(max_len=max_len),
            "disturbance": TimeSeriesBuffer(max_len=max_len),
            "throttle": TimeSeriesBuffer(max_len=max_len),
        }
        self.lock = threading.Lock()

        # 订阅向量类话题
        rospy.Subscriber("/uam_controller/debug/position", Vector3, self._vector_cb, "position")
        rospy.Subscriber("/uam_controller/debug/velocity", Vector3, self._vector_cb, "velocity")
        rospy.Subscriber("/uam_controller/debug/position_error", Vector3, self._vector_cb, "position_error")
        rospy.Subscriber("/uam_controller/debug/accel_cmd", Vector3, self._vector_cb, "accel_cmd")
        rospy.Subscriber("/uam_controller/debug/disturbance", Vector3, self._vector_cb, "disturbance")

        # 订阅四个电机油门
        for i in range(4):
            topic = f"/uam_controller/debug/throttle_{i}"
            rospy.Subscriber(topic, Float64, self._throttle_cb, i)

        plt.ion()
        self.fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=False)
        self.fig.suptitle("UAM 控制数据监控台")
        self.axes = axes.flatten()

        titles = [
            "位置 (m)",
            "速度 (m/s)",
            "位置误差 (m)",
            "加速度指令 (m/s²)",
            "扰动估计 (N)",
            "电机油门",
        ]
        for ax, title in zip(self.axes, titles):
            ax.set_title(title)
            ax.grid(True)

        self.lines: Dict[str, List[plt.Line2D]] = {}
        legend_labels = ["X", "Y", "Z"]
        color_map = ["tab:blue", "tab:orange", "tab:green"]
        for key, ax in zip([
            "position",
            "velocity",
            "position_error",
            "accel_cmd",
            "disturbance",
        ], self.axes[:5]):
            self.lines[key] = []
            for idx, color in enumerate(color_map):
                line, = ax.plot([], [], color=color, label=legend_labels[idx])
                self.lines[key].append(line)
            ax.legend(loc="upper right")

        # 电机油门单独处理
        ax_throttle = self.axes[5]
        ax_throttle.set_ylim(0.0, 1.1)
        ax_throttle.set_xlabel("时间 (s)")
        ax_throttle.set_ylabel("归一化油门")
        self.lines["throttle"] = []
        throttle_colors = ["tab:red", "tab:cyan", "tab:purple", "tab:brown"]
        for idx, color in enumerate(throttle_colors):
            line, = ax_throttle.plot([], [], color=color, label=f"Motor {idx}")
            self.lines["throttle"].append(line)
        ax_throttle.legend(loc="upper right")

    def _vector_cb(self, msg: Vector3, key: str):
        with self.lock:
            self.buffers[key].append(rospy.Time.now(), [msg.x, msg.y, msg.z])

    def _throttle_cb(self, msg: Float64, index: int):
        with self.lock:
            values = [0.0, 0.0, 0.0, 0.0]
            if self.buffers["throttle"].values:
                values = list(self.buffers["throttle"].values[-1])
            values[index] = msg.data
            self.buffers["throttle"].append(rospy.Time.now(), values)

    def _update_plot(self):
        with self.lock:
            for key, ax in zip([
                "position",
                "velocity",
                "position_error",
                "accel_cmd",
                "disturbance",
            ], self.axes[:5]):
                t, arr = self.buffers[key].as_arrays()
                if arr.size == 0:
                    continue
                for idx, line in enumerate(self.lines[key]):
                    line.set_data(t, arr[:, idx])
                ax.relim()
                ax.autoscale_view(True, True, True)

            # 油门
            t_throttle, arr_throttle = self.buffers["throttle"].as_arrays()
            if arr_throttle.size != 0:
                for idx, line in enumerate(self.lines["throttle"]):
                    line.set_data(t_throttle, arr_throttle[:, idx])
                ax_throttle = self.axes[5]
                ax_throttle.relim()
                ax_throttle.autoscale_view(True, True, True)
                ax_throttle.set_ylim(-0.05, 1.05)

        for ax in self.axes:
            ax.set_xlim(left=0.0)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.draw()
        plt.pause(0.001)

    def spin(self):
        rospy.loginfo("控制监控台已启动，等待调试话题数据...")
        rate = rospy.Rate(self.update_rate)
        plt.show(block=False)
        while not rospy.is_shutdown():
            self._update_plot()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


if __name__ == "__main__":
    try:
        monitor = ControlMonitor()
        monitor.spin()
    except rospy.ROSInterruptException:
        pass
