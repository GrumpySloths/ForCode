import mujoco_py
import os


def print_sites_positions(model):
    site_ids = [
        model.site_name2id(site_name) for site_name in model.site_names
    ]
    site_positions = [sim.data.site_xpos[site_id] for site_id in site_ids]
    for site_name, site_pos in zip(model.site_names, site_positions):
        print(f'{site_name}: {site_pos}')


choice = input("请输入文件路径所对应数字: ")
path = [
    "./front_leg_t1_exp.xml", "./front_leg_t1_large.xml", "rear_leg_t1_exp.xml"
]
# model = mujoco_py.load_model_from_path("./front_leg_t1_large.xml")
model = mujoco_py.load_model_from_path(path[int(choice)])
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
print_sites_positions(model)
while True:
    sim.step()
    viewer.render()