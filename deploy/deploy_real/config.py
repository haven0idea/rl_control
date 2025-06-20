from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)


            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            self.cmd_init = np.array(config["cmd_init"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            if  "frame_stack" in config:
                self.num_obs_frame=config["frame_stack"]
