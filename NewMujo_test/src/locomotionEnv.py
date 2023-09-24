import gymnasium as gym
import numpy as np

class GymEnv(gym.Env):

    def __init__(self, agent):
        self.agent = agent
        self.observation_space=gym.spaces.Box(low=-1,high=1,shape=(11,),dtypd=np.float32)
        self.action_space=gym.spaces.Box(low=-1,high=1,shape=(8,),dtype=np.float32)
    def reset(self, **kwargs):

        obs, info = self.agent.reset()

        return obs, info

    def step(self, action):
        self.agent.runStep(action)
        obs = []
        info = {}
        reward = 0.0
        terminated = False
        info["curFoot"] = self.agent.getFootWorldPosition_y()
        info["curFoot_z"] = self.agent.getFootWorldPosition_z()
        info["euler_z"], info["rot_mat"] = self.agent.getEuler_z()
        info["euler"]=self.agent.getEuler()
        info["slope_y"]=self.agent.getSlope_y()
        return obs, reward, terminated, False, info
