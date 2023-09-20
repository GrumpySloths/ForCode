import gymnasium as gym


class GymEnv(gym.Env):

    def __init__(self, agent):
        self.agent = agent

    def reset(self, **kwargs):

        obs, info = self.agent.reset()

        return obs, info

    def step(self, action):
        self.agent.runStep(action)
        obs = []
        info = {}
        reward = 0.0
        terminated = False
        info["curFoot"] = self.agent.getFootWorldPosition()
        info["euler_z"], info["rot_mat"] = self.agent.getEuler_z()
        info["euler"]=self.agent.getEuler()
        info["slope_y"]=self.agent.getSlope_y()
        return obs, reward, terminated, False, info
