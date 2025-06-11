import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import csv
import os
# 键盘控制
from pynput.keyboard import Key, Listener
import threading

#import onnx
import onnx
import onnxruntime as ort
import mujoco_viewer
from collections import deque
# ------------------------------------------------------
# 用于存储机器人的命令
command_lock = threading.Lock()
env_commands = [0, 0, 0, 0]  

# 更新命令
def update_command():
    with command_lock:
        print(f"当前机器人速度为: {env_commands[0]}, {env_commands[1]}, {env_commands[2]}, {env_commands[3]}")

# 键盘按键按下事件回调
def on_press(key):
    global env_commands
    try:
        if key.char == 'w':  # 按下 'w' 键时，向前移动
            env_commands[0] += 0.2
            update_command()

        elif key.char == 's':  # 按下 's' 键时，停止
            env_commands = [0, 0, 0, 0]
            print("当前机器人速度清零:", env_commands)

        elif key.char == 'x':  # 按下 'x' 键时，向后移动
            env_commands[0] -= 0.2
            update_command()

        elif key.char == 'a':  # 按下 'a' 键时，向左移动
            env_commands[1] += 0.2
            update_command()

        elif key.char == 'd':  # 按下 'd' 键时，向右移动
            env_commands[1] -= 0.2
            update_command()

        elif key.char == 'q':  # 按下 'q' 键时，向上移动
            env_commands[2] += 0.2
            update_command()

        elif key.char == 'e':  # 按下 'e' 键时，向下移动
            env_commands[2] -= 0.2
            update_command()

        elif key.char == 'z':  # 按下 'z' 键时，向左前方移动
            env_commands[3] += 0.2
            update_command()

        elif key.char == 'c':  # 按下 'c' 键时，向右前方移动
            env_commands[3] -= 0.2
            update_command()
        
    except AttributeError:
        pass  # 忽略特殊按键

# 监听键盘事件
def on_release(key):
    if key == Key.esc:  # 按下 'esc' 键退出
        return False
# 键盘监听函数，放到独立线程中
def listen_for_keyboard():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
# ------------------------------------------------------
# 转换数组为字符串
def array_to_str(arr):
    return np.array2string(arr, separator=',')[1:-1]  # 去掉开头和结尾的括号
     
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

simulation_data = []
# 处理数据并在仿真结束后批量写入文件
def save_simulation_data():
    with open(log_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'dof_pos_target', 'dof_pos_actual', 'dof_tau', 'base_pos', 'base_vel'])
        for row in simulation_data:
            writer.writerow(row)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["onnx_policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        log_name = config["log_name"]
        num_obs_frame = config["frame_stack"]

    # 启动键盘监听线程
    keyboard_thread = threading.Thread(target=listen_for_keyboard)
    keyboard_thread.start()
    simulation_time = 0.0
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, log_name)
    print(f"log_file_path: {file_path}")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    model = onnx.load(policy_path)
    onnx.checker.check_model(model)
    policy=ort.InferenceSession(policy_path)

    hist_obs = deque()
    for _ in range(num_obs_frame):
        hist_obs.append(np.zeros([1, num_obs], dtype=np.float32))

    viewer = mujoco_viewer.MujocoViewer(m, d)
    # 创建及覆盖原文件
    with open(log_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'dof_pos_target', 'dof_pos_actual', 'dof_tau', 'base_pos', 'base_vel'])
    # Close the viewer automatically after simulation_duration wall-seconds.
    start = time.time()
    try:
        while time.time() - start < simulation_duration:
            simulation_time += simulation_dt
            with command_lock:
                cmd[0] = env_commands[0]
                cmd[1] = env_commands[1]
                cmd[2] = env_commands[2]

            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega #3
                obs[3:6] = gravity_orientation #3
                obs[6:9] = cmd * cmd_scale #3
                obs[9 : 9 + num_actions] = qj #10
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj #10
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action #10

                obs_copy = np.clip(obs, -18, 18)
                hist_obs.append(obs_copy)
                hist_obs.popleft()

                policy_input = np.zeros([1,num_obs_frame*num_obs], dtype=np.float32)
                for i in range(num_obs_frame):
                    policy_input[0, i * num_obs : (i + 1) * num_obs] = hist_obs[i]

                obs_tensor = torch.from_numpy(policy_input)

                # policy inference
                outputs = policy.run(None, {'input': obs_tensor.numpy()}) 
                action = outputs[0]  # outputs 是一个包含所有输出的列表
                action = action = action.squeeze()
                action = np.clip(action, -4., 4.)
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                # print ("target:", target_dof_pos)
                # 将数据暂存到内存中
                simulation_data.append([simulation_time,
                                        array_to_str(target_dof_pos), 
                                        array_to_str(d.qpos[7:]), 
                                        array_to_str(tau), 
                                        array_to_str(d.qpos[:3]), 
                                        array_to_str(d.qvel[:3])])
            viewer.render()
            # Pick up changes to the physics state, apply perturbations, update options from GUI. 
            # viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    except KeyboardInterrupt:
        print("仿真中断，保存数据...")
    finally:
        # 确保仿真结束后保存数据
        save_simulation_data()
        keyboard_thread.join()
