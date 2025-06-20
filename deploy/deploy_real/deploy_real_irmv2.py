import lcm
from protocol.imu_lcmt import imu_lcmt
from protocol.robot_status_response_lcmt import robot_status_response_lcmt
from protocol.control_cmd_lcmt import control_cmd_lcmt
import threading
import time
import sys
import numpy as np
from config import Config
from collections import deque
import termios
import tty
import csv
import torch
from pynput.keyboard import Key, Listener
import os
# ------------------------------------------------------
# 用于存储机器人的命令
command_lock = threading.Lock()
global env_commands
env_commands = [0, 0, 0, '0']   
# 更新命令
def update_command():
    with command_lock:
        print(f"当前机器人速度和状态为: {env_commands[0]}, {env_commands[1]}, {env_commands[2]}, {env_commands[3]}")

# 键盘按键按下事件回调
def on_press(key):
    global env_commands
    try:
        if key.char == 'w':  # 按下 'w' 键时，向前移动
            env_commands[0] += 0.2
            update_command()

        elif key.char == 's':  # 按下 's' 键时，停止
            env_commands[0] = 0
            env_commands[1] = 0
            env_commands[2] = 0
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

        elif key.char == 'q':  # 按下 'q' 键时，左转
            env_commands[2] += 0.2
            update_command()

        elif key.char == 'e':  # 按下 'e' 键时，右转
            env_commands[2] -= 0.2
            update_command()

        elif key.char == '1':
            env_commands[3] = '1'
            update_command()

        elif key.char == '2':
            env_commands[3] = '2'
            update_command()

        elif key.char == '3':
            env_commands[3] = '3'
            update_command()

        elif key.char == '5':
            env_commands[3] = '5'
            update_command()
            
    except AttributeError:
        pass  # 忽略特殊按键

# 监听键盘事件
def on_release(key):
    if key == Key.esc:  # 按下 'esc' 键退出
        return False
stop_event = threading.Event()

# 键盘监听函数，放到独立线程中
def listen_for_keyboard():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        while not stop_event.is_set():
            listener.join(0.1)
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
log_name =  "data_irmv2.csv"
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

class MotorControlSDK:
    def __init__(self, config: Config):
        # Initialize LCM
        # 模型文件加载
        self.policy = torch.jit.load(config.policy_path)
        
        policy_input = np.zeros([1, config.num_obs_frame * config.num_obs], dtype=np.float32)
        # policy inference
        for _ in range(10):
            action_warmup= self.policy(torch.tensor(policy_input)).detach().numpy().squeeze()
        print("load successfully")

        self.config = config

        # 启动键盘监听线程
        self.keyboard_thread = threading.Thread(target=listen_for_keyboard)
        self.keyboard_thread.start()

        self.current_dir = os.getcwd()
        self.file_path = os.path.join(self.current_dir, log_name)
        print(f"log_file_path: {self.file_path}")
        
        self.lcm = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")

        # Subscribe to IMU and robot status topics
        self.lcm.subscribe("robot_status_imu", self.imu_callback)#udpm://239.255.76.67:7890?ttl=255
        self.lcm.subscribe("robot_status_response", self.robot_status_callback)#udpm://239.255.76.67:7670?ttl=255

        # LCM data holders
        self.imu_data = None
        self.robot_status_data = None

        # Mutex locks for thread-safe access
        self.imu_lock = threading.Lock()
        self.robot_status_lock = threading.Lock()

        # State
        self.state = "WAITING"  # Initial state
        self.nn_control_active = False
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = self.config.cmd_init
        self.hist_obs = deque()
        for _ in range(config.num_obs_frame):
            self.hist_obs.append(np.zeros([1, config.num_obs], dtype=np.float32))

        self.control_cmd = control_cmd_lcmt()
        self.buffer_action=np.zeros((2,10),dtype=np.float32)

    def imu_callback(self, channel, data):
        msg = imu_lcmt.decode(data)
        with self.imu_lock:
            self.imu_data = msg

    def robot_status_callback(self, channel, data):
        msg = robot_status_response_lcmt.decode(data)
        with self.robot_status_lock:
            self.robot_status_data = msg

    def lcm_listen(self):
        """Runs the LCM listener."""
        try:
            while True:
                self.lcm.handle()
        except KeyboardInterrupt:
            print("Shutting down LCM listener...")

    def publish_zero_torque(self):
        """Publishes zero torque commands."""
        control_cmd = control_cmd_lcmt()

        control_cmd.action_type = 8  # 1000  zero torque mode
        control_cmd.action_num = (self.config.num_actions +2)  # Example motor number
        control_cmd.action_position = [0.0] * (self.config.num_actions +2)
        control_cmd.action_velocity = [0.0] * (self.config.num_actions +2)
        control_cmd.action_acceleration = [0.0] * (self.config.num_actions +2)
        control_cmd.action_torque = [0.0] * (self.config.num_actions +2)
        self.lcm.publish("robot_control_cmd", control_cmd.encode())#udpm://239.255.76.67:7671?ttl=255
        print("pub zero torque!!!!!!!!!!!!")

    def publish_default_position(self):
        """Publishes default position commands."""
        control_cmd = control_cmd_lcmt()
        control_cmd.action_type = 12  # 1100  default pos mode
        control_cmd.action_num = (self.config.num_actions +2)
        control_cmd.action_position = [0.0] * (self.config.num_actions +2)
        control_cmd.action_velocity = [0.0] * (self.config.num_actions +2)
        control_cmd.action_acceleration = [0.0] * (self.config.num_actions +2)
        control_cmd.action_torque = [0.0] * (self.config.num_actions +2)
        self.lcm.publish("robot_control_cmd", control_cmd.encode())
        print("pub default pos!!!!!!!!!!!")

    def publish_damping_cmd(self):
        """Publishes zero torque commands."""
        control_cmd = control_cmd_lcmt()

        control_cmd.action_type = 15  # 1110  damping mode  only kd=8
        control_cmd.action_num = (self.config.num_actions +2)  # Example motor number
        control_cmd.action_position = [0.0] * (self.config.num_actions +2)
        control_cmd.action_velocity = [0.0] * (self.config.num_actions +2)
        control_cmd.action_acceleration = [0.0] * (self.config.num_actions +2)
        control_cmd.action_torque = [0.0] * (self.config.num_actions +2)
        self.lcm.publish("robot_control_cmd", control_cmd.encode())
        print("pub damping!!!!!!!!!!!")

    def process_and_publish(self):
        """Processes the IMU and robot status data, passes them through the NN, and publishes control commands."""

        # 将12维度电机信息转换成10自由度关节信息
        dof_idx=[0,1,2,3,4,6,7,8,9,10]
        start_time = time.time()
        while  self.nn_control_active:
            with self.imu_lock, self.robot_status_lock:
                if self.imu_data is None or self.robot_status_data is None:
                    print("未接收到完整的传感器信息！！！")
                    time.sleep(0.01)
                    continue

                count_time = time.time()
                with command_lock:
                    self.cmd[0] = env_commands[0]
                    self.cmd[1] = env_commands[1]
                    self.cmd[2] = env_commands[2]
                # Extract data for the neural network
                ang_vel = np.array(self.imu_data.estAngularRate) * self.config.ang_vel_scale
                gravityVec = np.array(self.imu_data.gravityVec)
                euler_angle = np.array(self.imu_data.estEulerAngle)

                qj_obs = np.array(self.robot_status_data.motor_pos) * self.config.dof_pos_scale
                dqj_obs = np.array(self.robot_status_data.motor_vel) * self.config.dof_vel_scale

                self.obs = np.zeros(config.num_obs, dtype=np.float32)
                self.obs[:3] = ang_vel
                self.obs[3:6] = gravityVec
                self.obs[6:9] = self.cmd * self.config.cmd_scale
                self.obs[9: 9+self.config.num_actions] = qj_obs[dof_idx]
                self.obs[9+self.config.num_actions: 9+2*self.config.num_actions] = dqj_obs[dof_idx]
                # 训练时action未乘scale
                self.obs[9+2*self.config.num_actions: 9+3*self.config.num_actions] = self.action

                obs_copy = np.clip(self.obs, -18, 18)

                self.hist_obs.append(obs_copy)
                self.hist_obs.popleft()

                policy_input = np.zeros([1, self.config.num_obs_frame * self.config.num_obs], dtype=np.float32)
                for i in range(self.config.num_obs_frame):
                    policy_input[0, i * self.config.num_obs: (i + 1) * self.config.num_obs] = self.hist_obs[i]

            # policy inference
            self.action = self.policy(torch.tensor(policy_input)).detach().numpy().squeeze()
            self.action = np.clip(self.action, -4., 4.)

            # # 平滑处理
            # self.buffer_action=np.roll(self.buffer_action,-1,axis=0)
            # self.buffer_action[-1]=self.action
            # self.action=self.buffer_action.mean(axis=0)

            # transform action to target_dof_pos
            target_dof_pos = self.config.default_angles + self.action * self.config.action_scale           
            
            pos_cmd=np.zeros(12)
            pos_cmd[dof_idx]=target_dof_pos
            tau = pd_control(target_dof_pos, qj_obs[dof_idx], self.kps, np.zeros_like(self.kds), dqj_obs[dof_idx], self.kds)

            time_now = time.time()
            simulation_data.append([time_now-start_time,
                        array_to_str(target_dof_pos), 
                        array_to_str(qj_obs[dof_idx]), 
                        array_to_str(tau), 
                        array_to_str(euler_angle), 
                        array_to_str(ang_vel)])
            
            # Create control command message
            self.control_cmd.action_type = 14 
            self.control_cmd.action_num = 12  # len(a_t)
            self.control_cmd.action_position = pos_cmd #target_dof_pos
            self.control_cmd.action_velocity = [0.0] * self.config.num_actions  # Placeholder
            self.control_cmd.action_acceleration = [0.0] * self.config.num_actions  # Placeholder
            self.control_cmd.action_torque = [0.0] * self.config.num_actions  # Placeholder

            # Publish control commands
            time_now=time.time()
            self.lcm.publish("robot_control_cmd", self.control_cmd.encode())
            print("pub_cmd_fr",time_now-count_time)
           
            time.sleep(0.02)  # Adjust loop frequency as needed

    def run(self):
        listener_thread = threading.Thread(target=self.lcm_listen, daemon=True)
        listener_thread.start()

        with open(log_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'dof_pos_target', 'dof_pos_actual', 'dof_tau', 'base_pos', 'base_vel'])

        print("Waiting for IMU and robot status data...")
        while True:
            with self.imu_lock, self.robot_status_lock:
                if self.imu_data is not None and self.robot_status_data is not None:
                    break
            time.sleep(0.1)

        print("IMU and robot status data received. Ready.")

        try:
            while True:
                if self.state == "WAITING":
                    print("Enter '1' to switch to Zero Torque state:")
                    with command_lock:
                        if env_commands[3] == '1':
                            self.state = "ZERO_TORQUE"

                if self.state == "ZERO_TORQUE":
                    print("Switching to Zero Torque state.")
                    self.publish_zero_torque()
                    print("Enter '2' to switch to Default Position state:")
                    with command_lock:
                        if env_commands[3] == "2":
                            self.state = "DEFAULT"

                if self.state == "DEFAULT":
                    print("Switching to Default Position state.")
                    self.publish_default_position()
                    print("Enter '3' to switch to Neural Network Control state:")
                    with command_lock:
                        if env_commands[3] == "3":
                            self.state = "NN_CONTROL"

                if self.state == "NN_CONTROL":
                    print("Switching to Neural Network Control state. Enter '5' to quit.")
                    self.nn_control_active = True
                    control_thread = threading.Thread(target=self.process_and_publish, daemon=True)
                    control_thread.start()

                    if True:
                        with command_lock:
                            if env_commands[3] == "5":
                                print("Exiting NN Control state and switching to Zero Torque.")
                                self.nn_control_active = False
                                control_thread.join()
                                self.state = "DAMPING_CMD"
                                self.publish_damping_cmd()

                if self.state == "DAMPING_CMD":
                    print("Preparing to exit...")
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, exiting gracefully...")

        finally:
            save_simulation_data()
            stop_event.set()
            self.keyboard_thread.join()
            print("Cleaned up and exiting.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1_ours.yaml")
    args = parser.parse_args()
    # Load config
    config_path = f"configs/{args.config}"
    config = Config(config_path)

    sdk = MotorControlSDK(config)
    # sdk.lcm_listen()  # Ensure the LCM listener is running
    sdk.run()
