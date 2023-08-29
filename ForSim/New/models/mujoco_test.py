import mujoco_py
import os

choice=input("请输入文件路径所对应数字: ")
path=["leg_fl_assets/fl_exp.xml","archive_t1_legs/front_leg_t1_large.xml","test.xml"]
# model = mujoco_py.load_model_from_path("./front_leg_t1_large.xml")
model = mujoco_py.load_model_from_path(path[int(choice)])
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
while True:
    sim.step()
    viewer.render()