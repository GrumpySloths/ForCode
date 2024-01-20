import numpy as np
import math
import sys

sys.path.append('/mnt/S58Data1/niujh/ForCode/NewMujo_test/src')
from LegModel.forPath import LegPath
# -----------------------------------------------------------
from LegModel.legs import LegModel
from LegModel.ETG_model import ETG_layer
from LegModel.ETG_model import ETG_model
#------------------------------------------------------------
ETG_T = 2
ETG_H = 20
PHASE = np.array([-np.pi / 2, 0])
ETG_T2 = 0.5
ETG_DT = 0.002
ETG_AMP = 0.2


#------------------------------------------------------------
class MouseController(object):
    """docstring for MouseController"""

    def __init__(self, fre, time_step, spine_angle, ETG_path=None):
        super(MouseController, self).__init__()
        PI = np.pi
        self.curStep = 0  # Spine
        self.ETG_T = 1 / fre
        self.ETG_DT = time_step
        self.ETG_agent = ETG_layer(self.ETG_T, self.ETG_DT, ETG_H, 0.04, PHASE,
                                   ETG_AMP, ETG_T2)
        self.ETG_model = ETG_model(ETG_path, self.ETG_agent)
        # Spine A = 0
        #self.turn_F = 0*PI/180
        #self.turn_H = 8*PI/180
        # Spine A = 20
        self.turn_F = 0 * PI / 180
        self.turn_H = 12 * PI / 180
        # self.turn_H = 0 * PI / 180

        self.pathStore = LegPath()
        # [LF, RF, LH, RH]
        # --------------------------------------------------------------------- #
        # self.phaseDiff = [0, PI, PI * 1 / 2, PI * 3 / 2]  # Walk
        #self.period = 3/2
        #self.SteNum = 36							#32 # Devide 2*PI to multiple steps
        #self.spinePhase = self.phaseDiff[3]
        # --------------------------------------------------------------------- #
        self.phaseDiff = [0, PI, PI, 0]  # Trot
        self.period = 2 / 2
        self.fre_cyc = fre  #1.25#0.80
        self.SteNum = int(1 / (time_step * self.fre_cyc))  #该变量意义是什么呢?
        print("----> ", self.SteNum)
        self.spinePhase = self.phaseDiff[3]
        # --------------------------------------------------------------------- #
        self.spine_A = 2 * spine_angle  #10 a_s = 2theta_s
        print("angle --> ", spine_angle)  #self.spine_A)
        self.spine_A = self.spine_A * PI / 180
        # --------------------------------------------------------------------- #
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
        self.fl_left = LegModel(leg_params)
        self.fl_right = LegModel(leg_params)
        self.hl_left = LegModel(leg_params)
        self.hl_right = LegModel(leg_params)
        # --------------------------------------------------------------------- #
        self.stepDiff = [0, 0, 0, 0]  #代表四条腿的相位差吗?
        for i in range(4):
            self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i] / (2 * PI))
        self.stepDiff.append(int(self.SteNum * self.spinePhase / (2 * PI)))
        print("self.stepDiff:", self.stepDiff)
        self.trgXList = [[], [], [], []]
        self.trgYList = [[], [], [], []]
        #记录front leg的ctrlData
        self.ctrlDatas_fl = [[], []]

    def reset(self):
        self.curStep = 0 
        self.trgXList = [[], [], [], []]
        self.trgYList = [[], [], [], []]
        #记录front leg的ctrlData
        self.ctrlDatas_fl = [[], []]

    def update(self, w, b):
        self.ETG_model.update(w, b)

    def getLegCtrl(self, leg_M, curStep, leg_ID):
        '''
		curStep: 代表着actuator摆动角度或相位
		'''

        def getETGPathPoint(curStep, leg_flag):
            t = curStep / self.SteNum * self.ETG_T
            # obs = self.ETG_agent.update(t)
            # pos = self.ETG_model.forward(obs)
            pos = self.ETG_model.forward2(t)
            if (leg_flag == 'F'):
                return pos
            else:
                return pos - 0.005

        curStep = curStep % self.SteNum
        turnAngle = self.turn_F
        leg_flag = "F"
        if leg_ID > 1:
            leg_flag = "H"
            turnAngle = self.turn_H

        # radian = 2 * np.pi * curStep / self.SteNum
        #currentPos = self.pathStore.getRectangle(radian, leg_flag)
        #下一行代码应该实现的是在给定radian下计算ankle距离actuator的理论相对距离
        # currentPos = self.pathStore.getOvalPathPoint(radian, leg_flag,
        #                                              self.period)
        currentPos = getETGPathPoint(curStep, leg_flag)
        trg_x = currentPos[0]
        trg_y = currentPos[1]
        self.trgXList[leg_ID].append(trg_x)
        self.trgYList[leg_ID].append(trg_y)
        #进一步转动角度了吗? 注意这里是一个旋转矩阵
        tX = math.cos(turnAngle) * trg_x - math.sin(turnAngle) * trg_y
        tY = math.cos(turnAngle) * trg_y + math.sin(turnAngle) * trg_x
        qVal = leg_M.pos_2_angle(tX, tY)
        return qVal

    def getSpineVal(self, spineStep):
        temp_step = int(spineStep)  # / 5)
        radian = 2 * np.pi * temp_step / self.SteNum
        return self.spine_A * math.cos(radian - self.spinePhase)
        #spinePhase = 2*np.pi*spineStep/self.SteNum
        #return self.spine_A*math.sin(spinePhase)

    def runStep(self):

        foreLeg_left_q = self.getLegCtrl(self.fl_left,
                                         self.curStep + self.stepDiff[0], 0)
        foreLeg_right_q = self.getLegCtrl(self.fl_right,
                                          self.curStep + self.stepDiff[1], 1)
        hindLeg_left_q = self.getLegCtrl(self.hl_left,
                                         self.curStep + self.stepDiff[2], 2)
        hindLeg_right_q = self.getLegCtrl(self.hl_right,
                                          self.curStep + self.stepDiff[3], 3)

        spineStep = self.curStep  #+ self.stepDiff[4]
        spine = self.getSpineVal(spineStep)
        #spine = 0
        self.curStep = (self.curStep + 1) % self.SteNum

        ctrlData = []

        # print("foreLeg_left_q --> ", foreLeg_left_q)

        #foreLeg_left_q = [1,0]
        #foreLeg_right_q = [1,0]
        #hindLeg_left_q = [-1,0]
        #hindLeg_right_q = [-1,0]
        ctrlData.extend(foreLeg_left_q)
        ctrlData.extend(foreLeg_right_q)
        ctrlData.extend(hindLeg_left_q)
        ctrlData.extend(hindLeg_right_q)
        self.ctrlDatas_fl[0].append(foreLeg_left_q[0] * 180 / np.pi)
        self.ctrlDatas_fl[1].append(foreLeg_left_q[1] * 180 / np.pi)
        # for i in range(3):
        #     ctrlData.append(0)
        # ctrlData.append(spine)
        # print(ctrlData)
        return ctrlData

    def getETGinfo(self, points):
        b = np.mean(points, axis=0)
        points_t = points - b

        obs = []
        # ts=radSample/(2*np.pi)*ETG_T
        ts = np.linspace(0, self.ETG_T, points.shape[0])
        for t in ts:
            v = self.ETG_agent.update(t)
            obs.append(v)
        obs = np.asarray(obs).reshape(-1, 20)
        w = np.linalg.pinv(obs).dot(points_t)

        return w, b
    def pointsCheck(self,points):
        '''做一个简单的测试验证points对应的轨迹是否合法'''
        for point in points:
            q1,q2=self.fl_left.pos_2_angle(*point)
            if q1<-0.5*np.pi or q1>0.5*np.pi:
                return False
            if q2<-0.5*np.pi or q2>0.5*np.pi:
                return False
        return True