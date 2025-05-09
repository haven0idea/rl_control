import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 读取 CSV 文件，指定分隔符为制表符或空格，具体视情况而定
def str_to_array(string):
    return np.array(ast.literal_eval(f'[{string}]'))

# 读取 CSV 文件
data = pd.read_csv('data_g1.csv', delimiter=",", header=0)
print(data.columns)

# 转换数据
simTime = data['time'].to_numpy(dtype=np.float32)
dof_pos_target = data['dof_pos_target'].apply(str_to_array)
dof_pos_actual = data['dof_pos_actual'].apply(str_to_array)
dof_tau = data['dof_tau'].apply(str_to_array)
base_pos = data['base_pos'].apply(str_to_array)
base_vel = data['base_vel'].apply(str_to_array)

# 绘图：电机目标位置和电机实际位置对比
def plot_motor_positions_and_actual():
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))  
    fig.suptitle("Motor Target vs Actual Positions (Degrees)")

    for i in range(12):  # 遍历 12 个电机
        ax = axes[i // 4, i % 4]
        # 绘制目标位置和实际位置的对比
        ax.plot(simTime, dof_pos_target.apply(lambda x: x[i]) * 180 / np.pi, 'b-', label='Target Position')
        ax.plot(simTime, dof_pos_actual.apply(lambda x: x[i]) * 180 / np.pi, 'g-', label='Actual Position')
        ax.set_title(f'Motor {i + 1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (deg)')
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

# 绘图：基座位置
def plot_base_position():
    plt.figure(figsize=(10, 6))
    plt.plot(simTime, base_pos.apply(lambda x: x[0]), label='X', color='r')
    plt.plot(simTime, base_pos.apply(lambda x: x[1]), label='Y', color='g')
    plt.plot(simTime, base_pos.apply(lambda x: x[2]), label='Z', color='b')
    plt.title("Base Position (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    plt.show()

# 绘图：基座速度
def plot_base_velocity():
    plt.figure(figsize=(10, 6))
    plt.plot(simTime, base_vel.apply(lambda x: x[0]), label='X Velocity', color='r')
    plt.plot(simTime, base_vel.apply(lambda x: x[1]), label='Y Velocity', color='g')
    plt.plot(simTime, base_vel.apply(lambda x: x[2]), label='Z Velocity', color='b')
    plt.title("Base Velocity (m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()
    plt.show()

# 绘图：电机力矩
def plot_motor_torques():
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))  
    fig.suptitle("Motor Torques (Nm)")

    for i in range(12):  # 遍历 12 个电机
        ax = axes[i // 4, i % 4]
        ax.plot(simTime, dof_tau.apply(lambda x: x[i]), 'b-', label='Torque')
        ax.set_title(f'Motor {i + 1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

# 调用绘图函数
plot_motor_positions_and_actual()
plot_base_position()
plot_base_velocity()
plot_motor_torques()
