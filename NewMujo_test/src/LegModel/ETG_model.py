import numpy as np
import os


class ETG_layer():

    def __init__(self, T, dt, H, sigma_sq, phase, amp, T2_radio):
        #T2_ratio mean the ratio forward t,在这里基本表示着周期的一半
        self.dt = dt
        self.T = T
        self.t = 0
        self.H = H
        self.sigma_sq = sigma_sq
        self.phase = phase
        self.amp = amp
        self.u = []
        self.omega = 2.0 * np.pi / T
        self.T2_ratio = T2_radio
        for h in range(H):
            t_now = h * self.T / (H - 0.9)
            self.u.append(self.forward(t_now))
        #self.u包含了u_ij的信息
        self.u = np.asarray(self.u).reshape(-1, 2)
        self.TD = 0

    def forward(self, t):
        x = []
        for i in range(self.phase.shape[0]):
            x.append(self.amp * np.sin(self.phase[i] + t * self.omega))
        return np.asarray(x).reshape(-1)

    def update(self, t=None):
        '''
        用于计算P=W*V+b中的V
        '''
        time = t if t is not None else self.t
        x = self.forward(time)
        self.t += self.dt
        r = []
        for i in range(self.H):
            dist = np.sum(np.power(x - self.u[i], 2)) / self.sigma_sq
            r.append(np.exp(-dist))
        r = np.asarray(r).reshape(-1)
        return r

    def update2(self, t=None, info=None):
        '''
        用于计算t时刻的V和t+T/2时刻的V，这里要计算不同时刻的V是为了得到常用的步态，即rf和lh足部同向，
        lf和rh足部同向且和另两条腿相差T/2 phase
        '''
        time = t if t is not None else self.t
        x = self.forward(time)
        x2 = self.forward(time + self.T2_ratio * self.T)
        self.t += self.dt
        r = []
        for i in range(self.H):
            dist = np.sum(np.power(x - self.u[i], 2)) / self.sigma_sq
            r.append(np.exp(-dist))
        r = np.asarray(r).reshape(-1)
        r2 = []
        for i in range(self.H):
            dist = np.sum(np.power(x2 - self.u[i], 2)) / self.sigma_sq
            r2.append(np.exp(-dist))
        r2 = np.asarray(r2).reshape(-1)
        return (r, r2)

    def observation_T(self):
        ts = np.arange(0, self.T, self.dt)
        x = {t: self.forward(t) for t in ts}
        r_all = {}
        for j in ts:
            r = []
            for i in range(self.H):
                dist = np.sum(np.power(x[j] - self.u[i], 2)) / self.sigma_sq
                r.append(np.exp(-dist))
            r_all[j] = np.asarray(r).reshape(-1)
        return r_all

    def get_phase(self):
        return self.forward(self.t - self.dt)

    def reset(self):
        self.t = 0
        self.TD = 0


class ETG_model():

    def __init__(self, ETG_path, ETG_agent) -> None:
        self.ETG_agent = ETG_agent
        if len(ETG_path) > 1 and os.path.exists(ETG_path):
            info = np.load(ETG_path)
            self.ETG_w = info['w']
            self.ETG_b = info['b']
        else:
            print("ETG_path is not existed")
    
    def update(self,w,b):
        self.ETG_w=w
        self.ETG_b=b

    def forward(self, x):
        '''
        计算P=WV+b中的P，P代表足末相对于电机的相对距离
        '''
        footPosition = x.dot(self.ETG_w) + self.ETG_b

        return footPosition

    def forward2(self, t):
        '''
        给定时间步t，计算相应时间的足末相对位置P
        '''
        obs = self.ETG_agent.update(t)
        pos = self.forward(obs)

        return pos
