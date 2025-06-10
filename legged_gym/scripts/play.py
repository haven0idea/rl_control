from pynput.keyboard import Key, Listener
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch

# 用于存储机器人的命令
env_commands = [0, 0, 0, 0]  

# 更新命令
def update_command():
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

# 启动监听器
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.continuous_push = False 
    env_cfg.domain_rand.randomize_base_mass = False 
    env_cfg.domain_rand.randomize_base_com = False 
    env_cfg.domain_rand.randomize_pd_gains = False 
    env_cfg.domain_rand.randomize_calculated_torque = False 
    env_cfg.domain_rand.randomize_link_mass = False 
    env_cfg.domain_rand.randomize_motor_zero_offset = False 
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.add_cmd_action_latency = False
    env_cfg.domain_rand.randomize_cmd_action_latency = False
    env_cfg.domain_rand.range_cmd_action_latency = [5, 5]
    env_cfg.domain_rand.add_obs_latency = False
    env_cfg.domain_rand.randomize_obs_motor_latency = False
    env_cfg.domain_rand.range_obs_motor_latency = [5, 5]
    env_cfg.domain_rand.randomize_obs_imu_latency = False
    env_cfg.domain_rand.range_obs_imu_latency = [5, 5]
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):

        env.commands[:,0] = env_commands[0]
        env.commands[:,1] = env_commands[1]
        env.commands[:,2] = env_commands[2]
        env.commands[:,3] = env_commands[3]

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

# 主函数
if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.num_envs = 10
    args.task = "irmv2"
    args.load_run = "/home/why/unitree_rl_gym/logs/irmv2"
    # args.task = "g1"
    # args.load_run = "/home/why/unitree_rl_gym/logs/g1"
    # 启动键盘监听
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()  # 启动监听器
    play(args)  
    listener.join()  # 等待监听器完成
