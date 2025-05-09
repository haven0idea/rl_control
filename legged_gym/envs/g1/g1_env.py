
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from legged_gym.utils.math import wrap_to_pi
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        # noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()
        # 减少对固定步态周期的依赖，几乎不使用
        # period = 0.8
        # offset = 0.5
        # self.phase = (self.episode_length_buf * self.dt) % period / period
        # self.phase_left = self.phase
        # self.phase_right = (self.phase + offset) % 1
        # self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        # print ("train_step!!!!!!!!!!!:", self.train_step)
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        # sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        # cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # sin_phase,
                                    # cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # sin_phase,
                                    # cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _resample_commands(self, env_ids):
        """ Randomly select commands of some environments, with curriculum learning """

        # 当前训练阶段（也可以直接使用 self.train_step 变量）
        iteration = self.train_step if hasattr(self, "train_step") else 0

        # 20% 静止命令，80% 移动命令
        rand_mask = torch.rand(len(env_ids), device=self.device) < 0.2  
        static_ids = env_ids[rand_mask]
        dynamic_ids = env_ids[~rand_mask]

        # 对静止命令的机器人，给它们静止命令
        self.commands[env_ids, 0] = 0
        self.commands[env_ids, 1] = 0
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = 0
        else:
            self.commands[env_ids, 2] = 0

        # # 对动态命令的机器人，给它们随机速度命令
        self.commands[dynamic_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(dynamic_ids), 1), device=self.device).squeeze(1)
        self.commands[dynamic_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(dynamic_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[dynamic_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(dynamic_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[dynamic_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(dynamic_ids), 1), device=self.device).squeeze(1)

        # 清理小速度，确保没有微小的运动命令
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > 0.1)


    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 判断是否为低速
        lin_speed = torch.norm(self.commands[:, :2], dim=1)
        ang_speed = torch.abs(self.commands[:, 2])
        total_speed = lin_speed + ang_speed
        low_speed = total_speed < 0.1

        # 计算当前接触腿数量
        contacts = (self.contact_forces[:, self.feet_indices, 2] > 1).float()
        num_contacts = contacts.sum(dim=1)
        # 行走：单脚接触地面
        gait_reward = (num_contacts == 1).float()
        # 站立：自由探索 双脚触地
        stand_reward = 1
        res = torch.where(low_speed, stand_reward, gait_reward)
        return res

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= (torch.norm(self.commands[:, :2], dim=1)+torch.abs(self.commands[:, 2])) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * ((torch.norm(self.commands[:, :2], dim=1)+torch.abs(self.commands[:, 2])) < 0.1)

    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_contact_no_vel(self):
        # Penalize contact with velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    