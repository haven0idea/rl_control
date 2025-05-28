import mujoco
import mujoco_viewer
from legged_gym import LEGGED_GYM_ROOT_DIR

xml_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/scene.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

for _ in range(10000000):
    mujoco.mj_step(model, data)
    viewer.render()