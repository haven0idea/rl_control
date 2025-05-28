from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class IRMV2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 45
        num_privileged_obs = 48 # 多了机身线速度
        num_actions = 12

    # class terrain(LeggedRobotCfg.terrain):
    #     # mesh_type = 'plane'
    #     mesh_type = 'trimesh'
    #     curriculum = False
    #     # rough terrain only:
    #     measure_heights = False
    #     static_friction = 0.6
    #     dynamic_friction = 0.6
    #     terrain_length = 8.
    #     terrain_width = 8.
    #     num_rows = 20  # number of terrain rows (levels)
    #     num_cols = 20  # number of terrain cols (types)
    #     max_init_terrain_level = 10  # starting curriculum state
    #     # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
    #     # terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
    #     terrain_proportions = [0, 0, 0, 0, 0, 0.5, 0.5]
    #     # 反弹
    #     restitution = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 摩擦
        randomize_friction = True
        friction_range = [0.1, 1.25]

        # 推力扰动
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        max_push_ang_vel = 0.6

        # 质量和质心
        randomize_base_mass = True
        added_base_mass_range = [-1.0, 3.0]
        randomize_base_com = True
        added_base_com_range = [-0.06, 0.06]

        # Link质量变化
        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]

        # PD参数
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]    

        # 扭矩
        randomize_calculated_torque = True
        torque_multiplier_range = [0.8, 1.2]

        # 电机零点偏移
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        # 关节摩擦、阻尼、转动惯量
        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]
        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]
        randomize_joint_armature = True
        joint_armature_range = [0.008, 0.06]    

        # 动作和观测延迟
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 10]
        add_obs_latency = True # no latency for obs_action
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 10]
        range_obs_imu_latency = [1, 10]

      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            collision = -1
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_air_time = 1  #1
            contact = 0.18
            stand_still = -1.0
            torques = -0.00001
class IRMV2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 3000
        run_name = ''
        experiment_name = 'g1'

# 最大迭代次数假设5000,在其中每次环境运行24次循环，在每次循环中得到了action，进行step，在step中进行降采样，对齐50Hz的时间线，返回一次的观测、奖励等
# 相当于每个环境用MC采样得到了24步数据，存到transition中，trade off 部分
# 底层仿真器是按照dt 0.005s即200Hz进行，但是仿真步长降采样decimation 4，训练按照0.02s即50Hz
# episode_length为20s，episode除以dt计算最大步长就是1000步，也就是1000次就重置一次
# command是10s重置一次

