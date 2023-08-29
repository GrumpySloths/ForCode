from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import time


class SimModel(object):
    """docstring for SimModel"""

    def __init__(
        self,
        modelPath,
    ):
        super(SimModel, self).__init__()
        self.model = load_model_from_path(modelPath)
        self.sim = MjSim(self.model)
        # print("init 1")
        self.viewer = MjViewer(self.sim)
        # print("init 2")
        self.viewer.cam.azimuth = 0
        self.viewer.cam.lookat[0] += 0.25
        self.viewer.cam.lookat[1] += -0.5
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.run_speed = 2

        self.sim_state = self.sim.get_state()
        self.sim.set_state(self.sim_state)
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
        self.movePath = [[], [], []]
        self.angle_AEF = ["leg_link_fl", "knee_down_fl", "ankle_fl"]
        self.angle_AEF_record = []

    def initializing(self):
        self.movePath = [[], [], []]
        self.legRealPoint_x = [[], [], [], []]
        self.legRealPoint_y = [[], [], [], []]

    def runStep(self, ctrlData, cur_time_step):
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
        step_num = int(cur_time_step / 0.002)
        # ctrlData确定不是设定为想要转过的角度值吗?
        self.sim.data.ctrl[:] = ctrlData
        for i in range(step_num):
            self.sim.step()
            self.viewer.render()

        # print("angle_AEF:", self.angle_AEF_compute() * 180 / np.pi)
        self.angle_AEF_record.append(self.angle_AEF_compute() * 180 / np.pi)
        # get_site_xpos获取的是site的全局坐标
        tData = self.sim.data.get_site_xpos(self.fixPoint)
        for i in range(3):
            self.movePath[i].append(tData[i])
        for i in range(4):
            originPoint = self.sim.data.get_site_xpos(self.legPosName[i][0])
            currentPoint = self.sim.data.get_site_xpos(self.legPosName[i][1])
            # print(originPoint, currentPoint)
            tX = currentPoint[1] - originPoint[1]
            tY = currentPoint[2] - originPoint[2]
            self.legRealPoint_x[i].append(tX)
            self.legRealPoint_y[i].append(tY)
            # if i == 0:
            #     print("ty_fl --> ", tY)

    def getTime(self):
        return self.sim.data.time

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
        A = np.array(self.sim.data.get_site_xpos(self.angle_AEF[0]))
        E = np.array(self.sim.data.get_site_xpos(self.angle_AEF[1]))
        F = np.array(self.sim.data.get_site_xpos(self.angle_AEF[2]))
        AE = np.linalg.norm(A - E)
        EF = np.linalg.norm(E - F)
        AF = np.linalg.norm(A - F)
        angle_AEF = self.LawOfCosines_angle(AE, EF, AF)
        return angle_AEF