import matplotlib.pyplot as plt
import numpy as np
import os

ETG_T = 2


def saveETGinfo(path, w, b, param):
    np.savez(path, w=w, b=b, param=param)


def ETG_trj_plot(w, b, ETG_agent, idx, outdir="./"):
    outdir = os.path.join(outdir, "plot_log")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, "ETG_trj_{}.png".format(idx))
    ts_new = np.linspace(0, ETG_T, 100)
    obs1 = []
    for t in ts_new:
        v = ETG_agent.update(t)
        obs1.append(v)
    obs1 = np.asarray(obs1).reshape(-1, 20)
    points_new = obs1.dot(w) + b
    plt.plot(points_new[:, 0], points_new[:, 1])
    plt.savefig(outpath)
    plt.close()


def infoRecord(theMouse, theController):
    # plt.plot(theMouse.angle_AEF_record)
    # # plt.show()
    # plt.plot(theController.ctrlDatas_fl[0], label='1eg_joint_fl')
    # plt.plot(theController.ctrlDatas_fl[1], label="thigh_joint_fl")
    # plt.legend()
    # plt.show()
    # dis = theMouse.drawPath()
    # theMouse.drawPath_3d()
    # print("py_v --> ", dis / timeCost)
    # print("sim_v --> ", dis / (run_steps_num * time_step))
    # theMouse.savePath("own_125")

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
                               theController.trgYList[i],
                               label="controller")
        axs[pos_1, pos_2].plot(theMouse.legRealPoint_x[i],
                               theMouse.legRealPoint_y[i],
                               label='real')
        axs[pos_1, pos_2].legend()

    plt.savefig("target_real_tragectory.png")
    plt.figure()
    plt.plot(theMouse.foot_z)
    plt.title("the foot_z of mouse")
    plt.savefig("foot_z_Mouse.png")
    plt.figure()

    # fig2, axs2 = plt.subplots(2, 2)
    subTitle2 = [
        "Fore Left linksite", "Fore Right linksite", "Hind Left linksite",
        "Hind Right linksite"
    ]
    for i in range(4):
        if (i % 2 == 0): continue
        plt.plot(theMouse.legLink_x[i],
                 theMouse.legLink_y[i],
                 label=subTitle2[i])
    plt.legend()
    # for i in range(4):
    #     pos_1 = int(i / 2)
    #     pos_2 = int(i % 2)
    #     print(pos_1, pos_2)
    #     axs2[pos_1, pos_2].set_title(subTitle[i])
    #     axs2[pos_1, pos_2].plot(theMouse.legLink_x[i], theMouse.legLink_y[i])

    plt.savefig("target_real_linksite.png")
    plt.figure()
    plt.plot(theMouse.FlRlAnkleDistance_x, theMouse.FlRlAnkleDistance_y)
    plt.title("FlRl ankle relative distance")
    plt.savefig("FlRlAnkleDistance")
    plt.figure()
    plt.plot(theMouse.FlRlLinkDistance_x, theMouse.FlRlLinkDistance_y)
    plt.title("FlRl Link relative distance")
    plt.savefig("FlRlLinkDistance")
