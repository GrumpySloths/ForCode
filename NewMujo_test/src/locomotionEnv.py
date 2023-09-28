import gymnasium as gym
from parl.utils import logger, summary
import numpy as np


class GymEnv(gym.Env):

    def __init__(self, agent, debug_stat=False):
        self.agent = agent
        self.steps = 0
        self.endFoot = 0
        self.curFoot = 0
        self.startFoot = 0
        self.dist = 0
        self.debug_stat = debug_stat
        self.last_base10 = np.zeros((10, 3))

    def reset(self, **kwargs):

        obs, info = self.agent.reset()
        self.steps = 0
        self.curFoot = info["curFoot"][1]
        self.debug("startFoot={}".format(self.curFoot))
        self.endFoot = 0
        self.startFoot = self.curFoot
        self.last_base10 = np.zeros((10, 3))

        return obs, info

    def step(self, action):
        # if self.steps == 0:
        #     print("action=", action)
        self.steps += 1
        self.agent.runStep(action)
        obs = []
        info = {}
        reward = 0.0
        info["curFoot"] = self.agent.getFootWorldPosition_y()
        info["curFoot_z"] = self.agent.getFootWorldPosition_z()
        info["euler_z"], info["rot_mat"] = self.agent.getEuler_z()
        info["euler"] = self.agent.getEuler()
        info["slope_y"] = self.agent.getSlope_y()
        info["curBody"] = self.agent.getBodyPosition()
        # self.dist = self.endFoot - self.curFoot
        terminated = self.terminated(info)

        return obs, reward, terminated, False, info

    def terminated(self, info):
        '''判断episode是否结束
        '''
        if self.steps % 100 == 0:
            self.last_base10[1:, :] = self.last_base10[:9, :]
            self.last_base10[0, :] = np.array(info['curFoot']).reshape(1, 3)
            base_std = np.sum(np.std(self.last_base10, axis=0))
            if (self.steps >= 1000 and base_std <= 0.02):
                self.debug("小鼠停滞不前,base_std={}".format(base_std))
                return True
            if (abs(info['curFoot'][0]) > 0.2):
                self.debug("小鼠沿x方向移动过远")
                return True
            if (info['curFoot'][1] > 0.6):
                self.debug("小鼠沿反方向运动")
                return True
        if self.steps % 1000 == 0:
            self.endFoot = info["curFoot"][1]
            self.dist = self.curFoot - self.endFoot
            angle_z = info["euler_z"]
            slope_y = info["slope_y"]
            euler = info["euler"]
            self.debug("the move distance of 1000 step:{}".format(self.dist))
            self.debug("angle_x:{},angle_y:{},angle_z:{}".format(
                euler[0], euler[1], euler[2]))
            #防止小鼠停滞不前或其朝向偏离设定方向过远
            if (abs(angle_z) > 0.3):
                #小鼠朝向偏离既定方向过远
                self.debug("小鼠朝向偏离既定方向过远")
                return True
            if (abs(self.dist) < 1e-2):
                self.debug("小鼠停滞不前")
                #小鼠停滞不前处理
                return True
            if (abs(self.endFoot - self.startFoot) > 0.5):
                logger.info(
                    "the y pos of slope:{},endFoot_x:{},endFoot_y:{},endFoot_z:{}"
                    .format(slope_y, info["curFoot"][0], self.endFoot,
                            info["curFoot"][2]))
            self.curFoot = self.endFoot

            return False

    def debug(self, info):
        if self.debug_stat:
            print(info)
