# 20250220 Wakkk Modified for ROS Noetic
import os

import mujoco 
import mujoco.viewer as viewer 
import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, Twist, Vector3
from std_msgs.msg import Float64MultiArray
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 系统参数
gravity = 9.8066
mass = 2.0
dt = 0.005  # dt = model.opt.timestep
Ct = 6.450e-6
Cm = 1.056e-7
max_rot_velocity = 1434.7 

# 全局缓冲
odom = Odometry()
joint_angle = Float64MultiArray()
mpc_positions = []   # mpc 发布轨迹
gen_positions = []   # 自己生成的轨迹


class ROSHandler:
    def __init__(self):
        rospy.init_node('quad_sim_ros')
        # 发布器
        self.odom_pub = rospy.Publisher('/quad/odometry', Odometry, queue_size=10)
        self.joint_pub = rospy.Publisher('/quad/joint_angles', Float64MultiArray, queue_size=10)
        self.gripper_cam_pub = rospy.Publisher('/gripper_cam/image_raw', Image, queue_size=1)
        self.gripper_depth_pub = rospy.Publisher('/gripper_cam/depth', Image, queue_size=1)

        # 订阅电机命令
        self.motor_sub = rospy.Subscriber('/quad/motor_cmd', Float64MultiArray, self.motor_cb)
        self.current_motor_cmd = [0.0] * 4

        # 订阅 joint_cmd —— 控制第1、2 两个关节
        self.joint_sub = rospy.Subscriber('/quad/joint_cmd', Float64MultiArray, self.joint_cb)
        self.current_joint_cmd = [0.0, 0.0]

        # 新增 joint3_cmd —— 单独控制第3 个关节
        self.joint3_sub = rospy.Subscriber('/quad/joint3_cmd', Float64MultiArray, self.joint3_cb)
        self.current_joint_cmd3 = 0.0

        # 订阅 MPC 和 自生成轨迹
        self.path_sub = rospy.Subscriber('/mpc_trajectories', Path, self.path_cb)
        self.gen_path_sub = rospy.Subscriber('/generated_trajectory', Path, self.gen_path_cb)

        # 夹爪控制
        self.finger_sub = rospy.Subscriber('/quad/finger_cmd', Float64MultiArray, self.finger_cb)
        self.finger_cmd = 0.0

        # 图像桥
        self.bridge = CvBridge()

    def motor_cb(self, msg):
        # 四个电机输入裁剪到 [0,1]
        self.current_motor_cmd = list(np.clip(msg.data, 0.0, 1.0))

    def joint_cb(self, msg):
        # joint_cmd 只取前两个值
        if len(msg.data) >= 2:
            self.current_joint_cmd = list(msg.data[:2])

    def joint3_cb(self, msg):
        # joint3_cmd 只取第一个值
        if len(msg.data) > 0:
            self.current_joint_cmd3 = msg.data[0]

    def path_cb(self, msg):
        global mpc_positions
        mpc_positions = []
        for pose in msg.poses:
            p = pose.pose.position
            mpc_positions.append(np.array([p.x, p.y, p.z]))

    def gen_path_cb(self, msg):
        global gen_positions
        gen_positions = [np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
                         for p in msg.poses]

    def finger_cb(self, msg):
        if len(msg.data) > 0:
            self.finger_cmd = msg.data[0]

    def publish_odometry(self, d):
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        # 位置
        odom.pose.pose.position.x = d.qpos[0]
        odom.pose.pose.position.y = d.qpos[1]
        odom.pose.pose.position.z = d.qpos[2]
        # 姿态四元数，MuJoCo: [w,x,y,z] → ROS: [x,y,z,w]
        odom.pose.pose.orientation.x = d.sensordata[7]
        odom.pose.pose.orientation.y = d.sensordata[8]
        odom.pose.pose.orientation.z = d.sensordata[9]
        odom.pose.pose.orientation.w = d.sensordata[6]
        # 速度
        odom.twist.twist.linear = Vector3(d.qvel[0], d.qvel[1], d.qvel[2])
        odom.twist.twist.angular = Vector3(d.qvel[3], d.qvel[4], d.qvel[5])
        self.odom_pub.publish(odom)

    def publish_joint_angel(self, d):
        # 这里根据你实际的 qpos 索引调整
        joint_angle.data = [d.qpos[11], d.qpos[12], d.qpos[13]]
        self.joint_pub.publish(joint_angle)

    def publish_gripper_image(self, model, data):
        width, height = 640, 480
        camera_name = 'gripper_cam'

        if not hasattr(self, 'renderer'):
            self.renderer = mujoco.Renderer(model, width=width, height=height)

        # === 获取 RGB 图像 ===
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=camera_name)
        rgb = self.renderer.render()

        img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "gripper_cam_frame"
        #self.gripper_cam_pub.publish(img_msg)

        # === 获取 Depth 图像 ===
        # self.renderer.enable_depth_rendering()
        # self.renderer.update_scene(data, camera=camera_name)
        # depth = self.renderer.render()  # (H, W), float32, meters

        # # 发布为灰度图：仅用于可视化
        # depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)
        # depth_uint8 = (depth_norm * 255).astype(np.uint8)

        # depth_msg = self.bridge.cv2_to_imgmsg(depth_uint8, encoding='mono8')
        # depth_msg.header.stamp = rospy.Time.now()
        # depth_msg.header.frame_id = "gripper_cam_frame"
        #self.gripper_depth_pub.publish(depth_msg)

        self.renderer.disable_depth_rendering()

        # ✅ 可选：保存为 .npy 供点云用
        # np.save("/tmp/depth_map.npy", depth)


ros_handler = ROSHandler()
log_count = 0  # 调试用计数


def draw_mpc_trajectory(sim_viewer, traj1, traj2, color1=[0.5, 0, 0, 0.5], color2=[0, 0, 1, 0.5], width=0.005):
    sim_viewer.user_scn.ngeom = 0
    def draw_line(points, color):
        for i in range(len(points)-1):
            mujoco.mjv_connector(
                sim_viewer.user_scn.geoms[sim_viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=width,
                from_=points[i],
                to=points[i+1]
            )
            sim_viewer.user_scn.geoms[sim_viewer.user_scn.ngeom].rgba = color
            sim_viewer.user_scn.ngeom += 1

    draw_line(traj1, color1)
    draw_line(traj2, color2)


def control_callback(model, data):
    global log_count
    # 发布里程计和关节位置
    ros_handler.publish_odometry(data)
    ros_handler.publish_joint_angel(data)

    # 电机控制
    for i in range(4):
         data.actuator(f'motor{i+1}').ctrl[0] = ros_handler.current_motor_cmd[i]
    # 反算角速度并写回
    thrusts = [max(cmd, 0.0) * 12.5 for cmd in ros_handler.current_motor_cmd]
    motor_omegas = [np.sqrt(T / Ct) for T in thrusts]
    data.actuator('fl_motor').ctrl[0] = motor_omegas[0]
    data.actuator('fr_motor').ctrl[0] = motor_omegas[1]
    data.actuator('bl_motor').ctrl[0] = motor_omegas[2]
    data.actuator('br_motor').ctrl[0] = motor_omegas[3]

    # 前两个关节
    for j in range(2):
        data.actuator(f'act_joint{j+1}').ctrl[0] = ros_handler.current_joint_cmd[j]
    # 第三个关节
    data.actuator('act_joint3').ctrl[0] = ros_handler.current_joint_cmd3

    # 夹爪
    data.actuator('act_finger').ctrl[0] = ros_handler.finger_cmd

    # 调试打印
    log_count += 1
    if log_count >= 200:
        print("\n=== ODOMETRY DEBUG ===")
        print(f"Pos: {odom.pose.pose.position}")
        print(f"Ori: {odom.pose.pose.orientation}")
        print(f"Vel: {odom.twist.twist.linear}, AngVel: {odom.twist.twist.angular}")
        print(f"JointAngles: {joint_angle.data}")
        log_count = 0


if __name__ == '__main__':
    scene_path = os.path.join(os.path.dirname(__file__), 'scene', 'scene_quad_with_gripper.xml')
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.set_mjcb_control(control_callback)

    with viewer.launch_passive(model, data) as sim_viewer:
        while sim_viewer.is_running() and not rospy.is_shutdown():
            t0 = time.time()
            mujoco.mj_step(model, data)

            with sim_viewer.lock():
                draw_mpc_trajectory(sim_viewer, mpc_positions, gen_positions)
                ros_handler.publish_gripper_image(model, data)

            sim_viewer.sync()
            to_sleep = dt - (time.time() - t0)
            if to_sleep > 0:
                time.sleep(to_sleep)
