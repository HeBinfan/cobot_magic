#coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt


#全局变量
DT = 0.02   #时间步长   #50Hz
# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]  #6个关节
STATE_NAMES = JOINT_NAMES + ["gripper"] #状态名称列表，包括6个关节和夹爪
BASE_STATE_NAMES = ["linear_vel", "angular_vel"]    #底座状态名称列表


#加载HDF5数据集
def load_hdf5(dataset_dir, dataset_name):
    #首先检查数据集文件是否存在
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()
    #接着读取HDF5文件中的数据
    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        base_action = root['/base_action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if compressed:
            compress_len = root['/compress_len'][()]
    #如果图像数据是压缩的，则进行解压缩处理
    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list
    #最后返回读取的数据
    return qpos, qvel, effort, action, base_action, image_dict

#用于可视化数据集中的单个episode
def main(args):
    #解析输入参数
    dataset_dir = args['dataset_dir']   #数据集目录
    episode_idx = args['episode_idx']   #episode索引
    task_name   = args['task_name'] #任务名称
    dataset_name = f'episode_{episode_idx}' #数据集名称
    #加载HDF5数据集
    qpos, qvel, effort, action, base_action, image_dict = load_hdf5(os.path.join(dataset_dir, task_name), dataset_name)
    
    print('hdf5 loaded!!')
    #保存视频文件
    save_videos(image_dict, action, DT,  video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4')) #video_path:视频保存路径为dataset_dir/dataset_name_video.mp4
   

    #可视化关节状态和底座动作
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png')) #plot_path:图像保存路径为dataset_dir/dataset_name_qpos.png
    visualize_base(base_action, plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png')) #plot_path:图像保存路径为dataset_dir/dataset_name_base_action.png


#用于保存视频文件
def save_videos(video, actions, dt, video_path=None):
    # 将多个摄像头的视频拼接在一起并保存为一个视频文件
    cam_names = list(video.keys())  #获取摄像头名称列表
    all_cam_videos = [] #用于存储所有摄像头的视频数据的列表
    for cam_name in cam_names:
        all_cam_videos.append(video[cam_name])  #每个摄像头的视频数据
    all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension   #将多个摄像头的视频在宽度维度上拼接在一起
    # 保存视频文件
    n_frames, h, w, _ = all_cam_videos.shape    #获取视频的帧数、高度和宽度
    fps = int(1 / dt)   #计算视频的帧率
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) #定义视频写入对象
    for t in range(n_frames):
        image = all_cam_videos[t]   #获取当前帧的图像数据
        image = image[:, :, [2, 1, 0]]  # swap B and R channel  #将图像的BGR通道转换为RGB通道
        cv2.imshow("images",image)  #显示图像
        cv2.waitKey(30) #等待30毫秒
        print("episode_id: ", t, "left: ", np.round(actions[t][:7], 3), "right: ", np.round(actions[t][7:], 3), "\n")   #打印当前帧的动作信息
        out.write(image)    #将图像写入视频文件
    out.release()   #释放视频写入对象
    print(f'Saved video to: {video_path}')  #打印视频保存路径

#用于可视化关节状态
def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:  #如果提供了覆盖标签,调用时未提供则使用默认标签
        label1, label2 = label_overwrite    #覆盖标签
    else:
        label1, label2 = 'State', 'Command'   #默认标签

    qpos = np.array(qpos_list) # ts, dim    #将关节位置列表转换为NumPy数组
    command = np.array(command_list)    # ts, dim   #将关节命令列表转换为NumPy数组
    
    num_ts, num_dim = qpos.shape    #获取时间步数和维度数
    h, w = 2, num_dim   #每个子图的高度和宽度
    num_figs = num_dim  #子图数量等于维度数
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))  #创建子图

    # plot joint state  #绘制关节状态
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]    #状态名称列表，包括左右侧
    for dim_idx in range(num_dim):  #遍历每个维度
        ax = axs[dim_idx]   #获取当前子图
        ax.plot(qpos[:, dim_idx], label=label1, color='orangered')  #绘制关节位置曲线
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')  #设置子图标题
        ax.legend()  #显示图例

    # plot arm command
    # for dim_idx in range(num_dim):
    #     ax = axs[dim_idx]
    #     ax.plot(command[:, dim_idx], label=label2)
    #     ax.legend()

    if ylim:    #如果提供了y轴范围
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()  #调整子图布局
    plt.savefig(plot_path)  #保存图像文件
    print(f'Saved qpos plot to: {plot_path}')   #打印图像保存路径
    plt.close()  #关闭图像

#用于可视化底座动作
def visualize_base(readings, plot_path=None):
    readings = np.array(readings) # ts, dim  #将底座动作列表转换为NumPy数组
    num_ts, num_dim = readings.shape    #获取时间步数和维度数
    num_figs = num_dim  #子图数量等于维度数
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))  #创建子图

    # plot joint state
    all_names = BASE_STATE_NAMES    #底座状态名称列表
    for dim_idx in range(num_dim):  #遍历每个维度
        ax = axs[dim_idx]   #获取当前子图
        ax.plot(readings[:, dim_idx], label='raw')  #绘制原始底座动作曲线
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')    #绘制平滑后的底座动作曲线
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')    #绘制平滑后的底座动作曲线
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')   #绘制平滑后的底座动作曲线
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')  #设置子图标题
        ax.legend() #显示图例


    plt.tight_layout()  #调整子图布局
    plt.savefig(plot_path)      #保存图像文件
    print(f'Saved effort plot to: {plot_path}')  #打印图像保存路径
    plt.close() #关闭图像

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #创建参数解析器
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)  #数据集目录
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', 
                        default="aloha_mobile_dummy", required=False)   #任务名称
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',default=0, required=False) #episode索引
    #解析命令行参数并调用主函数
    main(vars(parser.parse_args())) #调用主函数
