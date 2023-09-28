import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import time


class SimModel(object):
    """docstring for SimModel"""

    def __init__(self, modelPath, timeStep, freFrame, render):
        super(SimModel, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(modelPath)
        self.model.opt.timestep = timeStep
        self.time_step = timeStep
        self.freFrame = freFrame
        self.render = render
        print("opt.time:", self.model.opt.timestep)
        self.data = mujoco.MjData(self.model)
        # print("init 1")
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # self.viewer2 = mujoco_viewer.MujocoViewer(self.model, self.data)
            # print("init 2")
            self.viewer.cam.azimuth = 0
            # self.viewer.cam.lookat[0] += 0.25
            # self.viewer.cam.lookat[1] += -0.5
            self.viewer.cam.distance = self.model.stat.extent * 0.5
            self.viewer.run_speed = 0.1
        # self.legPosName = [["router_shoulder_fl", "foot_s_fl"],
        #                    ["router_shoulder_fr", "foot_s_fr"],
        #                    ["router_hip_rl", "foot_s_rl"],
        #                    ["router_hip_rr", "foot_s_rr"]]
        self.legPosName = [["leg_link_fl", "ankle_fl"],
                           ["leg_link_fr", "ankle_fr"],
                           ["leg_link_rl", "ankle_rl"],
                           ["leg_link_rr", "ankle_rr"]]
        self.fixPoint = "body_ss"  #"neck_ss"
        self.legRealPoint_x = [[], [], [], []]
        self.legRealPoint_y = [[], [], [], []]
        self.legLink_x = [[], [], [], []]
        self.legLink_y = [[], [], [], []]
        self.legLink_z=[[], [], [], []]
        self.movePath = [[], [], []]
        self.angle_AEF = ["leg_link_fl", "knee_down_fl", "ankle_fl"]
        self.angle_AEF_record = []
        self.FlRlLinkDistance_x = []
        self.FlRlLinkDistance_y = []
        self.FlRlAnkleDistance_x = []
        self.FlRlAnkleDistance_y = []
        self.foot_z = []

        self.paused = False


    def initializing(self):
        self.movePath = [[], [], []]
        self.legRealPoint_x = [[], [], [], []]
        self.legRealPoint_y = [[], [], [], []]
        self.legLink_x = [[], [], [], []]
        self.legLink_y = [[], [], [], []]
        self.legLink_z=[[], [], [], []]
        self.angle_AEF_record = []
        self.FlRlLinkDistance_x = []
        self.FlRlLinkDistance_y = []
        self.FlRlAnkleDistance_x = []
        self.FlRlAnkleDistance_y = []
        self.foot_z = []

    def runStep(self, ctrlData):
        # ------------------------------------------ #
        # ID 0, 1 left-fore leg and coil
        # ID 2, 3 right-fore leg and coil
        # ID 4, 5 left-hide leg and coil
        # ID 6, 7 right-hide leg and coil
        # Note: For leg, it has [-1: front; 1: back]
        # Note: For fore coil, it has [-1: leg up; 1: leg down]
        # Note: For hide coil, it has [-1: leg down; 1: leg up]
        # ------------------------------------------ #
        # ID 08 is neck		(Horizontal)
        # ID 09 is head		(vertical)
        # ID 10 is spine	(Horizontal)  [-1: right, 1: left]
        # Note: range is [-1, 1]
        # ------------------------------------------ #
        # step_num = int(cur_time_step / self.model.opt.timestep)
        # ctrlData确定不是设定为想要转过的角度值吗?
        if ctrlData.shape[0]==12:
            self.data.ctrl[:] = ctrlData
        else:
            self.data.ctrl[:8]=ctrlData

        # print("self.data.qacc:", self.data.qacc)
        # for i in range(step_num):
        if self.render:
            step_start = time.time()
            mujoco.mj_step(self.model, self.data)
            # mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            # print("step time:", time.time() - step_start)
            time_until_next_step = (
                self.model.opt.timestep) / self.freFrame - (time.time() -
                                                            step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        else:
            mujoco.mj_step(self.model, self.data)
        # print("angle_AEF:", self.angle_AEF_compute() * 180 / np.pi)
        self.angle_AEF_record.append(self.angle_AEF_compute() * 180 / np.pi)
        # get_site_xpos获取的是site的全局坐标
        tData_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     self.fixPoint)
        tData = self.data.site_xpos[tData_id]
        # print(tData)
        # tData = self.sim.data.get_site_xpos(self.fixPoint)
        for i in range(3):
            self.movePath[i].append(tData[i])
        FlLink = []
        FlAnkle = []
        RlLink = []
        RlAnkle = []

        foot_z_temp = 0
        for i in range(4):
            legPosName2id_0 = mujoco.mj_name2id(self.model,
                                                mujoco.mjtObj.mjOBJ_SITE,
                                                self.legPosName[i][0])
            legPosName2id_1 = mujoco.mj_name2id(self.model,
                                                mujoco.mjtObj.mjOBJ_SITE,
                                                self.legPosName[i][1])
            originPoint = self.data.site_xpos[legPosName2id_0]
            currentPoint = self.data.site_xpos[legPosName2id_1]
            if (i == 0):
                FlLink = originPoint
                FlAnkle = currentPoint
            if (i == 2):
                RlLink = originPoint
                RlAnkle = currentPoint
            # print(currentPoint)
            # print(originPoint, currentPoint)
            self.legLink_x[i].append(currentPoint[0])
            self.legLink_y[i].append(currentPoint[1])
            self.legLink_z[i].append(currentPoint[2])
            tX = currentPoint[1] - originPoint[1]
            tY = currentPoint[2] - originPoint[2]
            foot_z_temp += tY
            self.legRealPoint_x[i].append(tX)
            self.legRealPoint_y[i].append(tY)
            # if i == 0:
            #     print("ty_fl --> ", tY)
        self.foot_z.append(foot_z_temp / 4)
        # self.FlRlLinkDistance_x.append(FlLink[1] - RlLink[1])
        # self.FlRlLinkDistance_y.append(FlLink[2] - RlLink[2])
        # self.FlRlAnkleDistance_x.append(FlAnkle[1] - FlAnkle[1])
        # self.FlRlAnkleDistance_y.append(FlAnkle[2] - RlAnkle[2])
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # 该循环的作用是什么呢?是防止起步摔倒一直在做准备吗?
        for i in range(100):
            #ctrlData = 0
            # 应该是用于设置actuator的相位信息
            ctrlData = np.array([0, 1, 0, 1, 0.0, 1, 0.0, 1, 0, 0, 0, 0])
            self.runStep(ctrlData)
        # print("first stage")
        curFoot = self.getFootWorldPosition_y()
        curFoot_z = self.getFootWorldPosition_z()
        self.initializing()
        info = {}
        info["curFoot"] = curFoot
        info["curFoot_z"] = curFoot_z
        info["curBody"]=self.getBodyPosition()
        info["euler_z"], info["rot_mat"] = self.getEuler_z()
        info["euler"]=self.getEuler()

        obs = np.zeros(11)
        obs[:8]=ctrlData[:8]
        obs[8:]=info["euler"]
        
        return obs, info

    def getTime(self):
        return self.data.time

    def point_distance_line(self, point, line_point1, line_point2):
        #计算向量
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(
            vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance

    def drawPath_3d(self):
        x = self.movePath[0]
        y = self.movePath[1]
        z = self.movePath[2]

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)

        # Set the labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()

    def drawPath(self):
        #print(self.movePath)
        path_X = self.movePath[0]
        path_Y = self.movePath[1]
        tL = len(path_X)
        """
		order = 3
		tY = []
		for i in range(tL):
			tY.append(-path_Y[i])
		parameter = np.polyfit(tY, path_X, order)
		smooth_x = []
		for i in range(tL):
			tVal = 0
			for j in range(order):
				tVal = tVal + parameter[order] * tY[i] ** j
			smooth_x.append(tVal)
		dis = 0
		for i in range(tL-1):
			dX = smooth_x[i]-smooth_x[i+1]
			dY = path_Y[i]-path_Y[i+1]
			dis = dis + math.sqrt(dX*dX + dY*dY)
		print("Dis --> ", dis)
		"""

        ds = 1
        dL = int(tL / ds)
        check_x = []
        check_y = []
        print(tL)
        for i in range(dL):
            check_x.append(path_X[i * ds])
            check_y.append(path_Y[i * ds])

        check_x.append(path_X[-1])
        check_y.append(path_Y[-1])

        #dis = 50
        #for i in range(dL):
        #	dX = check_x[i]-check_x[i+1]
        #	dY = check_y[i]-check_y[i+1]
        #	dis = dis + math.sqrt(dX*dX + dY*dY)
        #dis = path_Y[0] - path_Y[-1]
        dX = path_X[0] - path_X[-1]
        dY = path_Y[0] - path_Y[-1]
        dis = math.sqrt(dX * dX + dY * dY)
        #dis = path_Y[0] - path_Y[-1]
        print("Dis --> ", dis)

        start_p = np.array([check_x[0], check_y[0]])
        end_p = np.array([check_x[-1], check_y[-1]])

        maxDis = 0
        for i in range(tL):
            cur_p = np.array([path_X[i], path_Y[i]])
            tDis = self.point_distance_line(cur_p, start_p, end_p)
            if tDis > maxDis:
                maxDis = tDis
        print("MaxDiff --> ", maxDis)
        plt.plot(path_X, path_Y)
        plt.plot(check_x, check_y)
        plt.grid()
        plt.show()

        return dis

    def savePath(self, flag):
        filePath = "Data/path_" + flag + ".txt"
        trajectoryFile = open(filePath, 'w')
        dL = len(self.movePath[0])
        for i in range(dL):
            for j in range(3):
                trajectoryFile.write(str(self.movePath[j][i]) + ' ')
            trajectoryFile.write('\n')
        trajectoryFile.close()

    def print_actuator(self):
        print("actuator_names:")
        print(self.model.actuator_names)

    def print_sites(self):
        print("site_names:")
        print(self.model.site_names)

    def print_sites_positions(self):
        site_ids = [
            self.model.site_name2id(site_name)
            for site_name in self.model.site_names
        ]
        site_positions = [
            self.sim.data.site_xpos[site_id] for site_id in site_ids
        ]
        for site_name, site_pos in zip(self.model.site_names, site_positions):
            print(f'{site_name}: {site_pos}')

    def LawOfCosines_angle(self, la, lb, lc):
        angle_ab_cos = (la * la + lb * lb - lc * lc) / (2 * la * lb)
        #print("----> ", angle_ab_cos)
        if abs(angle_ab_cos) > 1:
            return -10
        angle_ab = math.acos(angle_ab_cos)
        return angle_ab

    def angle_AEF_compute(self):
        site_A_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                      self.angle_AEF[0])
        site_E_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                      self.angle_AEF[1])
        site_F_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                      self.angle_AEF[2])
        A = np.array(self.data.site_xpos[site_A_id])
        E = np.array(self.data.site_xpos[site_E_id])
        F = np.array(self.data.site_xpos[site_F_id])
        AE = np.linalg.norm(A - E)
        EF = np.linalg.norm(E - F)
        AF = np.linalg.norm(A - F)
        angle_AEF = self.LawOfCosines_angle(AE, EF, AF)
        return angle_AEF

    def getFootWorldPosition_y(self):
        '''
        获取小鼠足末位置的世界坐标
        '''
        return (self.legLink_x[0][-1],self.legLink_y[0][-1],self.legLink_z[0][-1])
    

    def getFootWorldPosition_z(self):
        '''
        获取小鼠足末位置的世界坐标
        '''
        return self.legLink_y[0][-1]

    def getBodyPosition(self):
        '''获取小鼠body_ss的世界坐标'''
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "body_ss")
        pos = self.data.site_xpos[id]

        return pos
    
    def getEuler_z(self):
        '''
        获取XYZ欧拉变换后沿Z方向转过的角度
        '''
        # id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
        #                        "mouse_body")
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "body_ss")
        rot = self.data.site_xmat[id]
        angle_z = math.atan2(-rot[1], rot[0])

        return angle_z, rot
    
    def getEuler(self):
        '''
        获取XYZ欧拉变换后沿各轴所转过的角度
        '''
        euler=np.zeros(3)
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "body_ss")
        rot = self.data.site_xmat[id]
        euler[0]=math.atan2(-rot[5],rot[8])
        euler[1]=math.atan2(rot[2],math.sqrt(rot[1]**2+rot[0]**2))
        euler[2] = math.atan2(-rot[1], rot[0])

        return euler
    def getSlope_y(self):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "slope1")
        pos_y = self.data.geom_xpos[id][1]

        return pos_y
