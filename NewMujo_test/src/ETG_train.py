from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
from locomotionEnv import GymEnv
import time
import numpy as np
import utility
from alg.ETG_alg import SimpleGA
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

sys.path.append("..")
from parl.utils import logger, summary

# --------------------
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本文件所在的目录
script_directory = os.path.dirname(script_path)
project_path = "/".join(script_directory.split("/")[:-2])

DEBUG = True  #用于debug打印信息
ETG_PATH = os.path.join(script_directory, 'data/ETG_models/Slope_ETG.npz')
ROBOT_PATH = os.path.join(project_path, "ForSim/New/models/dynamic_4l.xml")
RUN_TIME_LENGTH = 8
SIGMA = 0.02
SIGMA_DECAY = 0.99
POP_SIZE = 40
ES_TRAIN_STEPS = 200
EVAL = True
EXP_ID = 3
#_______训练Reward配置_______________
VEL_D_BODY = 0.075  #理想情况下希望小鼠body所能达到的速度
VEL_D_FOOT = 0.085  #理想情况下希望小鼠foot所能达到的速度
REWARD_P = 5  #reward的增益效果，用于扩大或减小reward

Param_Dict = {
    'torso': 1.0,
    'up': 0.3,
    'feet': 0.2,
    'tau': 0.1,
    'done': 1,
    'velx': 0,
    'badfoot': 0.1,
    'footcontact': 0.1
}


#______________________________
def debug(info):
    if DEBUG:
        print(info)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=int, default=0, help="Evaluate or Not")
    parser.add_argument("--debug", type=int, default=0, help="debug or Not")
    parser.add_argument("--exp_id", type=int, default=4, help="用于设置输出文件的id号")
    parser.add_argument("--eval_ModelID", type=int, default=60, help="Evaluate时选取的模型序号")

    return parser

def Reward(info):
    '''用于计算小鼠运动的reward
    '''

    def c_prec(v, t, m):
        if m < 1e-5:
            print(m)
        w = np.arctanh(np.sqrt(0.95)) / m
        return np.tanh(np.power((v - t) * w, 2))

    re_vel_body = 1 - c_prec(min(info["vel_body"], VEL_D_BODY), VEL_D_BODY,
                             0.1)  #奖励小鼠body处于想要的速度
    re_vel_foot = 1 - c_prec(min(info["vel_foot"], VEL_D_FOOT), VEL_D_FOOT,
                             0.1)  #奖励小鼠foot处于想要的速度
    re_vel_foot *= Param_Dict["feet"]
    re_yaw = 1 - c_prec(info["euler_z"], 0, 0.5)  #奖励小鼠处于想要的yaw角度
    debug("vel_body={},raw={},pitch={},yaw={},vel_foot={}".format(
        info["vel_body"], info["euler"][0], info["euler"][1], info["euler"][2],
        info["vel_foot"]))
    reward = (re_vel_body + re_vel_foot) * re_yaw * REWARD_P

    return reward


def run_EStrain_episode(theController, env, maxStep):
    _, info = env.reset()
    step = 0
    curbody = info["curBody"][1]
    endbody = info["curBody"][1]
    curFoot = info["footPositions"][:, 1]
    endFoot = info["footPositions"][:, 1]
    Reward_rl = 0
    terminated = False
    while not terminated:
        step += 1
        tCtrlData = theController.runStep()  # No Spine
        #tCtrlData = theController.runStep_spine()		# With Spine
        ctrlData = np.asarray(tCtrlData)
        obs, reward, terminated, _, info = env.step(ctrlData)
        # if step % 20 == 0:
        endbody = info["curBody"][1]
        endFoot = info["footPositions"][:, 1]
        info['vel_body'] = (curbody - endbody) / (1 * 0.005)
        vel_foot = 0
        for i in range(4):
            vel_foot += (curFoot[i] - endFoot[i]) / (1 * 0.005) * 0.25
        info['vel_foot'] = vel_foot
        curReward = Reward(info)
        Reward_rl += curReward
        debug("curReward={},vel_body={},vel_foot={}".format(
            curReward, info['vel_body'], vel_foot))
        # debug("foot_z_mean={},angle_x={},angle_y={},angle_z={}".format(
        #     info["curFoot_z_mean"], info["euler"][0], info["euler"][1],
        #     info["euler_z"]))
        curbody = endbody
        curFoot = endFoot
        if step > maxStep:
            break
    logger.info("Final Reward_rl={}".format(Reward_rl))
    episode_reward = abs(env.endFoot - env.startFoot)
    return episode_reward, env.steps


if __name__ == '__main__':

    # logger.info('args:{}'.format(args))
    #_______
    parser=make_parser()
    args = parser.parse_args()
    render = True  #控制是否进行画面渲染
    fre_frame = 5  #画面帧率控制或者说小鼠运动速度控制
    fre = 0.5
    time_step = 0.005
    spine_angle = 0  #20
    run_steps_num = int(RUN_TIME_LENGTH / time_step)

    theMouse = SimModel(ROBOT_PATH, time_step, fre_frame, render=render)
    theController = MouseController(fre, time_step, spine_angle, ETG_PATH)
    info = np.load(ETG_PATH)
    prior_points = info["param"]
    ETG_param_init = prior_points.reshape(-1)

    ES_solver = SimpleGA(
        ETG_param_init.shape[0],
        sigma_init=SIGMA,
        sigma_decay=SIGMA_DECAY,
        sigma_limit=0.005,
        elite_ratio=0.1,
        weight_decay=0.005,
        popsize=POP_SIZE,
    )
    env = GymEnv(theMouse, debug_stat=DEBUG)

    if not args.eval:
        #____________ETG配置_______________

        #输出配置
        outdir = "./train_log/exp{}".format(EXP_ID)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logger.set_dir(outdir)
        logger.info("ETG train start:")
        for ei in range(ES_TRAIN_STEPS):
            start = time.time()
            solutions = ES_solver.ask()
            fitness_list = []
            steps = []
            for id, solution in enumerate(solutions):
                points_add = solution.reshape(-1, 2)
                new_points = prior_points + points_add
                print("new points.shape:",new_points.shape)
                w, b = theController.getETGinfo(new_points)
                theController.update(w, b)
                episode_reward, step = run_EStrain_episode(
                    theController, env, 8000)
                theController.reset()
                steps.append(step)
                # logger.info("%d th ES_train,%d solution :episode reward:%f" %
                #             (ei, id, episode_reward))
                fitness_list.append(episode_reward)
            results = ES_solver.result()
            sig = np.mean(results[3])
            fitness_list = np.asarray(fitness_list).reshape(-1)
            ES_solver.tell(fitness_list)
            end = time.time()
            logger.info(
                'ESSteps: {} Reward: {} step: {}  sigma:{},time:{}'.format(
                    ei + 1, np.max(fitness_list), np.mean(steps), sig,
                    end - start))
            summary.add_scalar('ES/episode_reward', np.mean(fitness_list),
                               ei + 1)
            summary.add_scalar('ES/episode_minre', np.min(fitness_list),
                               ei + 1)
            summary.add_scalar('ES/episode_maxre', np.max(fitness_list),
                               ei + 1)
            summary.add_scalar('ES/episode_restd', np.std(fitness_list),
                               ei + 1)
            summary.add_scalar('ES/episode_length', np.mean(steps), ei + 1)
            summary.add_scalar('ES/sigma', sig, ei + 1)
            if ei % 20 == 0:
                best_param = ES_solver.get_best_param()
                points_add = best_param.reshape(-1, 2)
                new_points = prior_points + points_add
                w_best, b_best = theController.getETGinfo(new_points)
                path = os.path.join(script_directory,
                                    "data/exp{}_ETG_models".format(EXP_ID))
                if not os.path.exists(path):
                    os.makedirs(path)
                path = os.path.join(path, "slopeBest_{}.npz".format(ei))
                # path = os.path.join(script_directory,"data/ETG_models/exp3/slopeBest_{}.npz".format(ei))
                theController.update(w_best, b_best)
                episode_reward, step = run_EStrain_episode(
                    theController, env, 8000)
                theController.reset()
                logger.info('Evaluation Reward: {} step: {} '.format(
                    episode_reward, step))
                summary.add_scalar('EVAL/episode_reward', episode_reward,
                                   ei + 1)
                utility.saveETGinfo(path, w_best, b_best, new_points)
                utility.ETG_trj_plot(w_best, b_best, theController.ETG_agent,
                                     ei, outdir)

    elif args.eval == True:
        print("start eval")
        # idx = 180
        ETG_Evalpath = os.path.join(
            script_directory,
            "data/exp{}_ETG_models/slopeBest_{}.npz".format(args.exp_id, args.eval_ModelID))
        # ETG_Evalpath = os.path.join(script_directory,
        #                             "data/ETG_models/Slope_ETG.npz")
        info = np.load(ETG_Evalpath)
        w = info["w"]
        b = info["b"]
        print("w.shape:", w.shape)
        print("b.shape:", b.shape)
        # points = info["param"]
        theController.update(w, b)
        utility.ETG_trj_plot(w, b, theController.ETG_agent, args.eval_ModelID)
        episode_reward, step = run_EStrain_episode(theController, env, 8000) #max_step 8000 default
        logger.info('Evaluation Reward: {} step: {} '.format(
            episode_reward, step))

    # 安全关闭模拟器
    if theMouse.render:
        theMouse.viewer.close()
    print("len of leg:",len(theMouse.legRealPoint_x))
    plt.scatter(theMouse.legRealPoint_x[0],theMouse.legRealPoint_y[0],label="real data")
    plt.legend()
    plt.savefig("realFootPath_ETG.png")
    # plt.show()
    # utility.infoRecord(theMouse, theController)
    