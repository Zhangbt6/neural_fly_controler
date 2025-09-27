#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线扰动估计节点：加载 Neural-Fly φ_net，执行 RLS 以估计 aerodynamic residual。"""
from __future__ import annotations

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerResponse

from pathlib import Path
import sys

# 访问 neural-fly Python 模块
PKG_ROOT = Path(__file__).resolve().parents[2] / 'neural-fly'
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))  # noqa: E402

from feature_utils import (AlphaFilter, compute_residual_force,
                           compute_total_thrust, quat_to_rot_matrix)  # noqa: E402
from mlmodel import load_model  # noqa: E402


class RLSAdapterNode:
    def __init__(self):
        rospy.init_node('rls_adapter', anonymous=False)

        # 话题配置
        self.odom_topic = rospy.get_param('~odom_topic', '/quad/odometry')
        self.motor_topic = rospy.get_param('~motor_cmd_topic', '/quad/motor_cmd')
        self.output_topic = rospy.get_param('~disturbance_topic', '/quad/disturbance_estimate')

        # 物理参数
        self.mass = rospy.get_param('~mass', 2.458)
        self.gravity = rospy.get_param('~gravity', 9.8066)
        self.motor_force_scale = rospy.get_param('~motor_force_scale', 12.5)
        self.acc_alpha = rospy.get_param('~acc_filter_alpha', 0.35)

        # RLS 参数
        self.lambda_ = rospy.get_param('~rls_lambda', 0.99)
        self.p0 = rospy.get_param('~p0', 1000.0)
        self.max_residual = rospy.get_param('~max_residual', 20.0)

        # 模型加载
        model_name = rospy.get_param('~model_name', 'phi_sim')
        model_folder = rospy.get_param('~model_folder', str(PKG_ROOT / 'models'))
        model_path = Path(model_folder) / f'{model_name}.pth'
        if not model_path.exists():
            raise RuntimeError(f'找不到模型 {model_path}, 请先运行 train_phi.py 训练并保存')
        self.model = load_model(model_name, modelfolder=str(model_folder) + '/')
        self.phi_net = self.model.phi.eval()
        torch.set_default_dtype(torch.double)

        self.dim_a = self.model.options['dim_a']
        self.dim_y = self.model.options['dim_y']

        self.A_hat = np.zeros((self.dim_a, self.dim_y))
        self.P = np.eye(self.dim_a) * self.p0

        self.acc_filter = AlphaFilter(self.acc_alpha)
        self.prev_time = None
        self.prev_vel = None
        self.last_motor_cmd = None

        self.pub = rospy.Publisher(self.output_topic, Vector3Stamped, queue_size=10)
        self.debug_pub = rospy.Publisher('/uam_controller/debug/disturbance_rls', Vector3Stamped, queue_size=10)

        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=50)
        self.motor_sub = rospy.Subscriber(self.motor_topic, Float64MultiArray, self._motor_cb, queue_size=50)

        self.reset_srv = rospy.Service('~reset', Trigger, self._reset_srv)
        rospy.loginfo('[rls_adapter] 已加载模型 %s，等待数据流...', model_path)

    # ------------------------------------------------------------------
    def _reset_internal_state(self):
        self.A_hat[:] = 0.0
        self.P[:] = np.eye(self.dim_a) * self.p0
        self.acc_filter.reset()
        self.prev_time = None
        self.prev_vel = None
        rospy.loginfo('[rls_adapter] RLS 状态已重置。')

    def _reset_srv(self, _req: Trigger):
        self._reset_internal_state()
        return TriggerResponse(success=True, message='reset done')

    def _motor_cb(self, msg: Float64MultiArray):
        data = np.array(msg.data, dtype=float)
        if data.shape[0] != 4:
            rospy.logwarn_throttle(5.0, '[rls_adapter] motor_cmd 长度非 4，忽略。')
            return
        self.last_motor_cmd = data

    def _odom_cb(self, msg: Odometry):
        if self.last_motor_cmd is None:
            return

        t = msg.header.stamp.to_sec()
        vel_world = np.array([msg.twist.twist.linear.x,
                              msg.twist.twist.linear.y,
                              msg.twist.twist.linear.z], dtype=float)
        pos = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z], dtype=float)
        quat_wxyz = np.array([msg.pose.pose.orientation.w,
                               msg.pose.pose.orientation.x,
                               msg.pose.pose.orientation.y,
                               msg.pose.pose.orientation.z], dtype=float)
        rot_bw = quat_to_rot_matrix(quat_wxyz)
        if self.prev_time is None:
            self.prev_time = t
            self.prev_vel = vel_world
            self.acc_filter.reset(np.zeros(3))
            return

        dt = t - self.prev_time
        if dt <= 1e-6:
            return

        raw_acc = (vel_world - self.prev_vel) / dt
        acc_world = self.acc_filter.update(raw_acc)

        total_thrust = compute_total_thrust(self.last_motor_cmd, self.motor_force_scale)
        residual_force = compute_residual_force(mass=self.mass,
                                                acc_world=acc_world,
                                                rot_bw=rot_bw,
                                                total_thrust=total_thrust,
                                                gravity=self.gravity)

        feature = np.hstack([vel_world, quat_wxyz, self.last_motor_cmd])
        phi = self._forward_phi(feature)
        d_hat = self._rls_update(phi, residual_force)
        self._publish_disturbance(d_hat, msg.header)

        self.prev_time = t
        self.prev_vel = vel_world

    # ------------------------------------------------------------------
    def _forward_phi(self, feature: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(feature[np.newaxis, :])
            phi = self.phi_net(x).cpu().numpy().squeeze()
        return phi

    def _rls_update(self, phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        phi = phi.reshape((-1, 1))  # (dim_a, 1)
        y = y.reshape((1, -1))      # (1, dim_y)

        denom = self.lambda_ + (phi.T @ self.P @ phi).item()
        K = (self.P @ phi) / denom               # (dim_a, 1)
        residual = y - (phi.T @ self.A_hat)     # (1, dim_y)
        self.A_hat += K @ residual
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_

        # 对称化避免数值漂移
        self.P = 0.5 * (self.P + self.P.T)

        d_hat = (phi.T @ self.A_hat).flatten()
        # 限幅，避免异常值冲击控制器
        norm = np.linalg.norm(d_hat)
        if norm > self.max_residual > 0:
            d_hat *= self.max_residual / norm
        return d_hat

    def _publish_disturbance(self, d_hat: np.ndarray, header):
        msg = Vector3Stamped()
        msg.header = header
        msg.vector.x, msg.vector.y, msg.vector.z = d_hat.tolist()
        self.pub.publish(msg)
        self.debug_pub.publish(msg)


def main():
    node = RLSAdapterNode()
    rospy.spin()


if __name__ == '__main__':
    main()
