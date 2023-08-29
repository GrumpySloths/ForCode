import mujoco_py
import os


# model = mujoco_py.load_model_from_path("./front_leg_t1_large.xml")
model = mujoco_py.load_model_from_path("./test.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
while True:
    sim.step()
    viewer.render()
