import gymnasium as gym
from parl.utils import logger, summary
import numpy as np

Param_Dict = {
    'feet':0.2,
    'body':1.0
}
VEL_D_BODY = 0.075  #理想情况下希望小鼠body所能达到的速度
VEL_D_FOOT = 0.085  #理想情况下希望小鼠foot所能达到的速度
REWARD_P = 5  #reward的增益效果，用于扩大或减小reward
OBS_VELOCITY=True #是否在observation 中添加小鼠body的速度

class GymEnv_RL(gym.Env):

    def __init__(self, agent, ETG_controller, debug_stat=False,foot_gain=0.2,body_gain=1.0):
        self.agent = agent
        self.ETG_controller = ETG_controller
        self.steps = 0
        self.endFoot = 0
        self.curFoot = 0
        self.startFoot = 0
        self.dist = 0
        self.debug_stat = debug_stat
        self.last_base10 = np.zeros((10, 3))
        self.foot_gain=foot_gain
        self.body_gain=body_gain

        if OBS_VELOCITY:
            logger.info("With Velocity,obs.shape=12")
            self.obs_shape=12
        else:
            logger.info("No Velocity,obs.shape=11")
            self.obs_shape=11
        self.action_space = gym.spaces.Box(-1, 1, shape=(8, ))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(self.obs_shape, ))
        self.steps_rl = 0
        self.last_ETG_act = np.zeros(8)

    def reset(self, **kwargs):
        if "ETG_w" in kwargs and "ETG_b" in kwargs:
            self.ETG_controller.update(kwargs["ETG_w"], kwargs["ETG_b"])

        self.ETG_controller.reset()
        self.last_ETG_act = self.ETG_controller.runStep()
        obs, info = self.agent.reset(next_ETG_act=self.last_ETG_act,obs_velocity=OBS_VELOCITY)
        self.steps = 0
        self.curFoot = info["curFoot"][1]
        self.curBody = info["curBody"][1]

        self.endFoot = 0
        self.endBody = 0
        self.startFoot = self.curFoot
        self.last_base10 = np.zeros((10, 3))

        return obs, info

    def step(self, action):
        '''action指的是强化学习的action'''

        self.steps_rl += 1
        start_body = 0
        end_body = 0  #记录小鼠 1 rl_step移动距离，即reward
        start_foot = np.zeros(4)
        end_foot = np.zeros(4)
        reward = 0.0
        obs = np.zeros(self.obs_shape)
        info = {}
        #20次循环是为了让小鼠的步幅和ETG_RL benchmark保持一致
        for i in range(20):
            self.steps += 1
            action_ETG = self.last_ETG_act
            ctrlData = action_ETG + action
            self.agent.runStep(ctrlData)

            info["curFoot"] = self.agent.getFootWorldPosition_y()
            info["curFoot_z"] = self.agent.getFootWorldPosition_z()
            info["euler_z"], info["rot_mat"] = self.agent.getEuler_z()
            info["euler"] = self.agent.getEuler()
            info["slope_y"] = self.agent.getSlope_y()
            info["curBody"] = self.agent.getBodyPosition()
            info["curFoot_z_mean"] = self.agent.getFootPosition_z()
            info["footPositions"] = self.agent.getFootWorldPositions()

            self.last_ETG_act = self.ETG_controller.runStep()  # No Spine

            if i == 0:
                start_body = info["curBody"][1]
                start_foot = info["footPositions"][:, 1]
            if i == 19:
                self.debug("foot_z_mean:{}".format(info["curFoot_z_mean"]))
                obs[:8] = self.last_ETG_act
                obs[8:11] = info["euler"]
                end_body = info["curBody"][1]
                end_foot = info["footPositions"][:, 1]
                info["vel_body"] = (start_body - end_body) / ((i + 1) * 0.005)
                if OBS_VELOCITY:
                    obs[11]=info["vel_body"]*10
                vel_foot = 0
                for i in range(4):
                    vel_foot += (start_foot[i] - end_foot[i]) / (
                        (i + 1) * 0.005) * 0.25
                info["vel_foot"] = vel_foot
            # self.dist = self.endFoot - self.curFoot
            terminated = self.terminated(info)
            if terminated:
                obs[:8] = self.last_ETG_act
                obs[8:11] = info["euler"]
                end_body = info["curBody"][1]
                end_foot = info["footPositions"][:, 1]
                info["vel_body"] = (start_body - end_body) / ((i + 1) * 0.005)
                if OBS_VELOCITY:
                    obs[11]=info["vel_body"]*10
                vel_foot = 0
                for i in range(4):
                    vel_foot += (start_foot[i] - end_foot[i]) / (
                        (i + 1) * 0.005) * 0.25
                info["vel_foot"] = vel_foot
                break

        reward,info = self.Reward(info)

        return obs, reward, terminated, False, info

    def Reward(self, info):
        '''用于计算小鼠运动的reward
        '''
        re_vel_body = 1 - self.c_prec(min(info["vel_body"], VEL_D_BODY),
                                      VEL_D_BODY, 0.1)  #奖励小鼠body处于想要的速度
        re_vel_foot = 1 - self.c_prec(info["vel_foot"],
                                      VEL_D_FOOT, 0.1)  #奖励小鼠foot处于想要的速度
        re_vel_foot*=self.foot_gain
        re_vel_body*=self.body_gain
        re_yaw = 1 - self.c_prec(info["euler_z"], 0, 0.5)  #奖励小鼠处于想要的yaw角度
        self.debug("vel_body={},yaw={},vel_foot={},Reward_body={},Reward_foot={}".format(
            info["vel_body"], info["euler"][2], info["vel_foot"],re_vel_body,re_vel_foot))
        
        info['feet']=re_vel_foot
        info['body']=re_vel_body
        reward = (re_vel_body + re_vel_foot) * re_yaw * REWARD_P
        
        return reward,info

    def terminated(self, info):
        '''判断episode是否结束
        '''
        if self.steps % 100 == 0:
            self.last_base10[1:, :] = self.last_base10[:9, :]
            self.last_base10[0, :] = np.array(info['curFoot']).reshape(1, 3)
            base_std = np.sum(np.std(self.last_base10, axis=0))
            if (self.steps >= 1000 and base_std <= 0.02):
                logger.info("小鼠停滞不前,base_std={}".format(base_std))
                return True
            if (abs(info['curFoot'][0]) > 0.2):
                logger.info("小鼠沿x方向移动过远")
                return True
            if (info['curFoot'][1] > 0.6):
                logger.info("小鼠沿反方向运动")
                return True
            if (info['curFoot_z_mean'] > 0):
                logger.info("小鼠摔倒了")
                return True
        if self.steps % 1000 == 0:
            self.endFoot = info["curFoot"][1]
            self.endBody = info["curBody"][1]
            self.dist = self.endFoot - self.curFoot
            self.bodyDist = self.endBody - self.curBody
            angle_z = info["euler_z"]
            slope_y = info["slope_y"]
            euler = info["euler"]
            self.debug("the move distance of 1000 step:{}".format(self.dist))
            self.debug("the body move distance of 1000 steps:{}".format(
                self.bodyDist))
            self.debug("angle_x:{},angle_y:{},angle_z:{}".format(
                euler[0], euler[1], euler[2]))
            #防止小鼠停滞不前或其朝向偏离设定方向过远
            if (abs(angle_z) > 0.3):
                #小鼠朝向偏离既定方向过远
                logger.info("小鼠朝向偏离既定方向过远")
                return True
            if (abs(self.dist) < 1e-2):
                logger.info("小鼠停滞不前")
                #小鼠停滞不前处理
                return True
            if (abs(self.endFoot - self.startFoot) > 0.5):
                logger.info(
                    "the y pos of slope:{},endFoot_x:{},endFoot_y:{},endFoot_z:{}"
                    .format(slope_y, info["curFoot"][0], self.endFoot,
                            info["curFoot"][2]))
            self.curFoot = self.endFoot
            self.curBody = self.endBody

        return False

    def debug(self, info):
        if self.debug_stat:
            print(info)

    def c_prec(self, v, t, m):
        if m < 1e-5:
            print(m)
        w = np.arctanh(np.sqrt(0.95)) / m
        return np.tanh(np.power((v - t) * w, 2))
