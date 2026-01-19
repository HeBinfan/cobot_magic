# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env

import collections
from collections import deque

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2

# 这个脚本用于在 ROS 系统下 同步采集多相机图像 + 机械臂关节状态 + （可选）深度图 
# + （可选）机器人底盘里程计，把一次 episode（若干时间步）的数据保存成 HDF5 文件，供训练/回放使用。
# 数据封装成 dm_env.TimeStep 风格的 observation，并将控制/主臂动作保存到 action。
# 主要流程：订阅话题 → 循环同步读取最新一组时刻的数据帧 → 将 observation/time-step 与对应 action 收集到内存 → 最后写入 HDF5。

# 保存数据函数
# 将内存中的 timesteps（长度为 T+1）和 actions（长度为 T）写入 HDF5 文件。
def save_data(args, timesteps, actions, dataset_path):
    # 数据字典
    data_size = len(actions)
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],  #关节位置 Joint position
        '/observations/qvel': [],  #关节速度 Joint velocity
        '/observations/effort': [],  #关节力 Effort
        '/action': [],             #实际发的动作  Action
        '/base_action': [],        # 机器人底盘速度 Base velocity
        # '/base_action_t265': [],
    }

    # 相机字典  观察的图像
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 奖励  前一帧

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])  #RGB图像
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])  #深度图

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        #
        root.attrs['sim'] = False   #是否仿真
        root.attrs['compress'] = False  #图像是否压缩

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations') #观测状态组
        image = obs.create_group('images')  #图像组
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), ) #每个相机创建一个数据集，存储图像数据
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                             chunks=(1, 480, 640), )

        _ = obs.create_dataset('qpos', (data_size, 14)) #关节位置
        _ = obs.create_dataset('qvel', (data_size, 14)) #关节速度
        _ = obs.create_dataset('effort', (data_size, 14))   #关节力
        _ = root.create_dataset('action', (data_size, 14))  #实际发的动作
        _ = root.create_dataset('base_action', (data_size, 2))  # 机器人底盘速度

        # data_dict write into h5py.File
        for name, array in data_dict.items():  
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)

#负责所有 ROS 订阅、帧同步与数据打包。
class RosOperator:
    # 使用多个 deque() 缓存不同话题接收到的消息（每种传感器一个 deque，默认手动管理大小上限）。
    # 初始化各个 deque，并创建 ROS 节点与订阅话题。
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.init()
        self.init_ros()  

    # 初始化各个 deque。
    # 初始化 CvBridge 和各个 deque。
    def init(self):
        self.bridge = CvBridge() # ROS图像与OpenCV图像转换 使用 CvBridge 将 ROS sensor_msgs/Image 转成 OpenCV 图像（numpy）。
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

    # 同步获取一组最新的传感器数据帧（多相机图像 + 机械臂关节状态 + 可选深度图 + 可选机器人底盘里程计）。
    # 作用：时间对齐（synchronization） —— 尝试从各个 deque 中取出时间戳不晚于当前最小帧时间 frame_time 的消息
    # , 目的是保证取到的一组消息来自同一时间点（或最近接近的时间点）。
    def get_frame(self):
        # 首先检查各个 deque 是否为空，若有任何一个为空则无法同步，返回 False。
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_left_deque) == 0 or self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_right_deque) == 0 or self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False    
        # 其次，从各个 deque 中取出时间戳不晚于 frame_time 的最新消息。
        # 彩色图像
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        # print("img_left:", img_left.shape)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        # 深度图像
        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        # 右深度图
        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time: 
                self.img_right_depth_deque.popleft()    #弹出早于frame_time的消息
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')    #深度图
        top, bottom, left, right = 40, 40, 0, 0 
        img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)   #填充边界

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_front_depth = cv2.copyMakeBorder(img_front_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        # 返回同步后的一组数据帧。
        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base)

    # 各个话题的回调函数，收到消息append后存入对应的 deque 中，超过 2000 条则弹出最早的一条。
    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    # 创建 ROS 节点并订阅话题（彩色图、深度图、左右主/从臂 JointState、/odom 等）。
    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True) 
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

    # 主处理函数，循环读取同步后的数据帧，打包成 dm_env.TimeStep 格式，收集 observation 与 action。
    def process(self):
        # 主处理函数，循环读取同步后的数据帧，打包成 dm_env.TimeStep 格式，收集 observation 与 action。
        timesteps = []
        actions = []
        # 1 初始化数据存储结构
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0
        
        # input_key = input("please input s:")
        # while input_key != 's' and not rospy.is_shutdown():
        #     input_key = input("please input s:")

        rate = rospy.Rate(self.args.frame_rate)
        print_flag = True
        # 2 主循环
        # 循环读取同步后的数据帧，打包成 dm_env.TimeStep 格式，收集 observation 与 action。
        while (count < self.args.max_timesteps + 1) and not rospy.is_shutdown():
            # 2 收集数据
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True
            count += 1
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result
            # 2.1 图像信息
            image_dict = dict()
            image_dict[self.args.camera_names[0]] = img_front
            image_dict[self.args.camera_names[1]] = img_left
            image_dict[self.args.camera_names[2]] = img_right

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            if self.args.use_depth_image:
                image_dict_depth = dict()
                image_dict_depth[self.args.camera_names[0]] = img_front_depth
                image_dict_depth[self.args.camera_names[1]] = img_left_depth
                image_dict_depth[self.args.camera_names[2]] = img_right_depth
                obs['images_depth'] = image_dict_depth
            obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            if self.args.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue
            # 中间帧 只包含mid
            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)
            # 保存时间步
            # 主臂保存状态
            action = np.concatenate((np.array(master_arm_left.position), np.array(master_arm_right.position)), axis=0)
            actions.append(action)
            timesteps.append(ts)
            print("Frame data: ", count)
            if rospy.is_shutdown():
                exit(-1)
            rate.sleep()
        # 返回采集到的 timesteps 和 actions。
        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions

# 解析命令行参数。
def get_arguments():
    # 参数解析
    parser = argparse.ArgumentParser()
    # 数据集存储路径
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)   #default用于设置默认值  #help用于帮助说明
    # 任务名称
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)   
    # episode索引
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)  #
    # 最大时间步长
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)
    # 相机名称列表
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    #  topic name of color image    #  彩色图像话题名称
    # 前置相机
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    # topic name of depth image
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    # topic name of arm #机械臂话题名称
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    # topic name of robot_base  #机器人底盘话题名称
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)
    # use robot base  #是否使用机器人底盘里程计
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    # collect depth image   #是否使用深度图
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
        # frame rate  #帧率
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    # 解析参数
    args = parser.parse_args()
    return args

# 主函数    # 主函数，解析参数，创建 RosOperator 实例，调用 process() 采集数据，最后调用 save_data() 保存数据。
def main():
    args = get_arguments()  #解析参数
    ros_operator = RosOperator(args)#创建 RosOperator 实例
    timesteps, actions = ros_operator.process() #调用 process() 采集数据
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)    #数据集存储路径
    # 检查数据长度是否满足要求
    if(len(actions) < args.max_timesteps):
        print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
        exit(-1)
# 创建数据集目录
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    # 数据集路径
    dataset_path = os.path.join(dataset_dir, "episode_" + str(args.episode_idx))
    # 保存数据
    save_data(args, timesteps, actions, dataset_path)


if __name__ == '__main__':
    main()

# python collect_data.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 0
