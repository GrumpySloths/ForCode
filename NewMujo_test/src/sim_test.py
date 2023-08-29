from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
from locomotionEnv import GymEnv
import time
import numpy as np
import utility
from alg.ETG_alg import SimpleGA
import os
# --------------------
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本文件所在的目录
script_directory = os.path.dirname(script_path)
project_path = "/".join(script_directory.split("/")[:-2])

ETG_PATH = os.path.join(script_directory, 'data/ETG_models/Slope_ETG.npz')
ROBOT_PATH = os.path.join(project_path, "ForSim/New/models/dynamic_4l.xml")
RUN_TIME_LENGTH = 8
SIGMA = 0.02
SIGMA_DECAY = 0.99
POP_SIZE = 40
ES_TRAIN_STEPS = 10


def run_EStrain_episode(theMouse, theController, env):
    obs, info = theMouse.reset()
    curFoot = info["curFoot"]
    endFoot = 0
    startFoot = curFoot
    terminated = False
    step = 0
    while not terminated:
        tCtrlData = theController.runStep()  # No Spine
        step += 1
        #tCtrlData = theController.runStep_spine()		# With Spine
        ctrlData = tCtrlData
        obs, reward, terminated, _, info = env.step(ctrlData)
        if step % 1000 == 0:
            endFoot = info["curFoot"]
            dist = endFoot - curFoot
            angle_z = info["euler_z"]
            # print("the move distance of 1000 step:", dist)
            # print("the euler of z axis:", angle_z)
            # print("rot_mat:", info["rot_mat"])
            # time.sleep(1)
            if (abs(dist) < 5e-4 or dist >= 0.01 or abs(angle_z) > 0.3):
                terminated = True
            curFoot = endFoot
    episode_reward = abs(endFoot - startFoot)
    return episode_reward


if __name__ == '__main__':
    render = True  #控制是否进行画面渲染
    fre_frame = 5  #画面帧率控制或者说小鼠运动速度控制
    fre = 0.5
    time_step = 0.002
    spine_angle = 0  #20
    run_steps_num = int(RUN_TIME_LENGTH / time_step)

    theMouse = SimModel(ROBOT_PATH, time_step, fre_frame, render=render)
    theController = MouseController(fre, time_step, spine_angle, ETG_PATH)
    env = GymEnv(theMouse)
    info = np.load(ETG_PATH)
    prior_points = info["param"]
    ETG_param_init = prior_points.reshape(-1)

    ES_solver = SimpleGA(
        ETG_param_init.shape[0],
        sigma_init=SIGMA,
        sigma_decay=SIGMA_DECAY,
        sigma_limit=0.005,
        elite_ratio=0.1,
        weight_decay=0.005,
        popsize=POP_SIZE,
    )
    for ei in range(ES_TRAIN_STEPS):
        solutions = ES_solver.ask()
        fitness_list = []
        for id, solution in enumerate(solutions):
            points_add = solution.reshape(-1, 2)
            new_points = prior_points + points_add
            w, b = theController.getETGinfo(new_points)
            theController.update(w, b)
            episode_reward = run_EStrain_episode(theMouse, theController, env)
            print("%d th ES_train,%d solution :episode reward:%f" %
                  (ei, id, episode_reward))
            fitness_list.append(episode_reward)
        ES_solver.tell(fitness_list)

    # 安全关闭模拟器
    if theMouse.render:
        theMouse.viewer.close()
    utility.infoRecord(theMouse, theController)
