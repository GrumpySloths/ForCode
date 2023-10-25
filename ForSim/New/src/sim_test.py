from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time

# --------------------
RUN_TIME_LENGTH = 50
if __name__ == '__main__':
    fre = 0.5
    time_step = 0.002
    spine_angle = 0  #20
    run_steps_num = int(RUN_TIME_LENGTH / time_step)
    # print("hello")
    theMouse = SimModel("../models/dynamic_4l.xml")
    # print("debug1")
    theController = MouseController(fre, time_step, spine_angle)
    #打印actuator信息
    theMouse.print_actuator()
    #打印site信息
    # theMouse.print_sites()
    #打印sites的位置信息
    # theMouse.print_sites_positions()

    # 该循环的作用是什么呢?是防止起步摔倒一直在做准备吗?
    for i in range(100):
        #ctrlData = 0
        # 应该是用于设置actuator的相位信息
        ctrlData = [0, 1, 0, 1, 0.0, 1, 0.0, 1, 0, 0, 0, 0]
        theMouse.runStep(ctrlData, time_step)
    print("first stage")
    theMouse.initializing()
    start = time.time()
    for i in range(run_steps_num):
        #print("Step --> ", i)
        tCtrlData = theController.runStep()  # No Spine
        #tCtrlData = theController.runStep_spine()		# With Spine
        ctrlData = tCtrlData
        theMouse.runStep(ctrlData, time_step)
    end = time.time()
    timeCost = end - start
    print("Time -> ", timeCost)

    plt.plot(theMouse.angle_AEF_record)
    # plt.show()
    plt.plot(theController.ctrlDatas_fl[0], label='1eg_joint_fl')
    plt.plot(theController.ctrlDatas_fl[1], label="thigh_joint_fl")
    plt.legend()
    plt.show()
    dis = theMouse.drawPath()
    theMouse.drawPath_3d()
    print("py_v --> ", dis / timeCost)
    print("sim_v --> ", dis / (run_steps_num * time_step))
    theMouse.savePath("own_125")

    #'''
    fig, axs = plt.subplots(2, 2)
    subTitle = [
        "Fore Left Leg", "Fore Right Leg", "Hind Left Leg", "Hind Right Leg"
    ]
    for i in range(4):
        pos_1 = int(i / 2)
        pos_2 = int(i % 2)
        print(pos_1, pos_2)
        axs[pos_1, pos_2].set_title(subTitle[i])
        axs[pos_1, pos_2].plot(theController.trgXList[i],
                               theController.trgYList[i])
        axs[pos_1, pos_2].plot(theMouse.legRealPoint_x[i],
                               theMouse.legRealPoint_y[i])

    plt.show()
    plt.plot(theController.trgXList[0],
             theController.trgYList[0],
             label='Target trajectory')
    plt.plot(theMouse.legRealPoint_x[0],
             theMouse.legRealPoint_y[0],
             label='Real trajectory ')
    plt.legend()
    plt.xlabel('y-coordinate (m)')
    plt.ylabel('z-coordinate (m)')
    plt.grid()
    plt.savefig("target_real_tragectory.png")
    #'''
