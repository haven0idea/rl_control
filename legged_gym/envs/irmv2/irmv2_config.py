from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class IRMV2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'Joint_hip_l_yaw': 0.,
            'Joint_hip_l_roll': 0.,
            'Joint_hip_l_pitch': -0.4363 ,  #18degree
            'Joint_knee_l_pitch': 0.7854,   #36degree
            'Joint_ankle_l_pitch': -0.3491,  #20degree
            'Joint_hip_r_yaw': 0.,
            'Joint_hip_r_roll': 0.,
            'Joint_hip_r_pitch': -0.4363 , 
            'Joint_knee_r_pitch': 0.7854,  
            'Joint_ankle_r_pitch': -0.3491,  
        }
    
    class env(LeggedRobotCfg.env):
        frame_stack = 10
        c_frame_stack = 5

        # num_observations = 45
        # num_privileged_obs = 48 # 多了机身线速度
        single_num_observations = 45 - 6
        num_observations = int(frame_stack * single_num_observations)
        single_num_privileged_obs = 48 - 6
        num_privileged_obs = int (c_frame_stack * single_num_privileged_obs)
        num_actions = 10

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

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 摩擦
        randomize_friction = True
        friction_range = [0.1, 1.25]

        # 推力扰动
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.3

        # 质量和质心
        randomize_base_mass = True
        added_base_mass_range = [-1.0, 1.0]
        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]

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

        # 对不同机器人影响很大，需要根据关节情况设定
        # 关节摩擦、阻尼、转动惯量
        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15] #[0.01, 1.15]
        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5] #[0.3, 1.5]
        randomize_joint_armature = True
        joint_armature_range = [0,0.005]    

        # 动作和观测延迟
        add_cmd_action_latency = False
        randomize_cmd_action_latency = False
        range_cmd_action_latency = [1, 2]
        add_obs_latency = False # no latency for obs_action
        randomize_obs_motor_latency = False
        randomize_obs_imu_latency = False
        range_obs_motor_latency = [1, 2]
        range_obs_imu_latency = [1, 2]
      
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_l_yaw':100,  
                     'hip_l_roll':100,
                     'hip_r_yaw':100,  
                     'hip_r_roll': 100,  
                     'hip_l_pitch':200,  
                     'hip_r_pitch': 200,  
                     'knee':200,  
                     'ankle': 20,  
                     }  # [N*m/rad]
        damping = { 'hip_l_yaw': 2,
                    'hip_l_roll':2,
                    'hip_r_yaw':2,
                    'hip_r_roll': 2,
                    'hip_l_pitch': 4,
                    'hip_r_pitch':4,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/irmv_v2/irmv_v2_0319.urdf'
        name = "irmv2"
        foot_name = "ankle"
        knee_name = "knee"
        penalize_contacts_on = ["hip","knee","body"]
        terminate_after_contacts_on = ["hip","knee","body"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.42
        min_dist = 0.15
        max_dist = 0.50
        max_contact_force = 200. # forces above this value are penalized
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0  #1.0
            tracking_ang_vel = 0.5  #0.5
            lin_vel_z = -1 #-1
            ang_vel_xy = -0.05 #-0.05
            orientation = -1.0
            base_height = 0.05 #0.05
            dof_acc = -2.5e-7 #-2.5e-7
            dof_vel = -1e-3 #-1e-3
            collision = 0
            dof_pos_limits = -5.0
            alive = 0.15 #0.15
            hip_pos = -1.0 #-1.0
            contact_no_vel = -0.2
            feet_air_time = 1  
            contact = 0.18
            stand_still = -1.0 #-1.0
            torques = -5e-6 #-0.00001
            action_smoothness = -0.003 #-0.003

            foot_slip = -0.1 #-0.1
            feet_swing_height = -20 #-20.0
            joint_error = -0.25 #-0.25
            feet_distance = 0.2 #0.2
            knee_distance = 0.2 #0.2
            energy_square = -1e-4
            track_vel_hard = 0.1 #0.5
            
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            quat=1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 4

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.02  
            dof_vel = 1.5  
            lin_vel = 0.1
            ang_vel = 0.5 
            gravity = 0.05
            quat=0.03+0.02
            height_measurements = 0.1

class IRMV2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef_start = 0.01
        entropy_coef_end = 0.0005
        entropy_coef_decay_iters = 5000
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        max_iterations = 5000
        run_name = ''
        experiment_name = 'irmv2'

        # resume = True
        # load_run = "Jun05_10-35-32_domain_randomization-entropy" # -1 = last run
        # checkpoint = 4000 # -1 = last saved model
        # resume_path = None # updated from load_run and chkpt

# 最大迭代次数假设5000,在其中每次环境运行24次循环，在每次循环中得到了action，进行step，在step中进行降采样，对齐50Hz的时间线，返回一次的观测、奖励等
# 相当于每个环境用MC采样得到了24步数据，存到transition中，trade off 部分
# 底层仿真器是按照dt 0.005s即200Hz进行，但是仿真步长降采样decimation 4，训练按照0.02s即50Hz
# episode_length为20s，episode除以dt计算最大步长就是1000步，也就是1000次就重置一次
# command是10s重置一次

