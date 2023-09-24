from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
from locomotionEnv import GymEnv
import time
import numpy as np
import utility
from alg.ETG_alg import SimpleGA
import os
from parl.utils import logger, summary,ReplayMemory
from model.mujoco_agent import MujocoAgent
from model.mujoco_model import MujocoModel
from alg.sac import SAC
# --------------------
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本文件所在的目录
script_directory = os.path.dirname(script_path)
project_path = "/".join(script_directory.split("/")[:-2])

DEBUG=True #用于debug打印信息
ETG_PATH = os.path.join(script_directory, 'data/ETG_models/Slope_ETG.npz')
ROBOT_PATH = os.path.join(project_path, "ForSim/New/models/dynamic_4l.xml")
RUN_TIME_LENGTH = 8
SIGMA = 0.02
SIGMA_DECAY = 0.99
POP_SIZE = 40
ES_TRAIN_STEPS = 200
EVAL = True
EXP_ID=4
MEMORY_SIZE = int(1e7)
MAX_STEPS=int(1e8) #训练的总步数
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ACT_BOUND_NOW=0.3   #determine the ratio of rl action
WARMUP_STEPS = 1e5
def debug(info):
    if DEBUG:
        print(info)
def run_EStrain_episode(theMouse, theController, env,rpm,action_bound):
    obs, info = theMouse.reset()
    curFoot = info["curFoot"][1]
    endFoot = 0
    startFoot = curFoot
    terminated = False
    step = 0
    critic_loss_list = []
    actor_loss_list = []
    while not terminated:
        action_ETG = theController.runStep()  # No Spine
        if rpm.size() < WARMUP_STEPS:
            action_rl = np.random.uniform(-1, 1, size=action_dim)
        else:
            action_rl = agent.sample(obs)
        step += 1
        #tCtrlData = theController.runStep_spine()		# With Spine
        ctrlData = action_ETG+action_rl
        '''
        这里要思考下reward该如何设置了
        '''
        obs, reward, terminated, _, info = env.step(ctrlData)
        if step%100==0:
            if(abs(info['curFoot'][0])>0.2):
                terminated=True
        if step % 1000 == 0:
            endFoot = info["curFoot"][1]
            # print("endFoot=", endFoot)
            dist = endFoot - curFoot
            angle_z = info["euler_z"]
            slope_y = info["slope_y"]
            euler=info["euler"]
            # debug("the move distance of 1000 step:{}".format(dist))
            debug("angle_x:{},angle_y:{},angle_z:{}".format(euler[0],euler[1],euler[2]))
            #防止小鼠停滞不前或其朝向偏离设定方向过远
            if (abs(angle_z) > 0.3):
                #小鼠朝向偏离既定方向过远
                debug("小鼠朝向偏离既定方向过远")
                terminated = True
            if (abs(dist) < 1e-2):
                debug("小鼠停滞不前")
                #小鼠停滞目前处理
                terminated = True
            if (abs(endFoot - startFoot) > 0.5):
                logger.info(
                    "the y pos of slope:{},endFoot_x:{},endFoot_y:{},endFoot_z:{}"
                    .format(slope_y, info["curFoot"][0], endFoot,
                            info["curFoot"][2]))
            curFoot = endFoot

    episode_reward = abs(endFoot - startFoot)
    return episode_reward, step


if __name__ == '__main__':
    
    # logger.info('args:{}'.format(args))
    #_______
    render = False  #控制是否进行画面渲染
    fre_frame = 5  #画面帧率控制或者说小鼠运动速度控制
    fre = 0.5
    time_step = 0.002
    spine_angle = 0  #20
    run_steps_num = int(RUN_TIME_LENGTH / time_step)

    theMouse = SimModel(ROBOT_PATH, time_step, fre_frame, render=render)
    theController = MouseController(fre, time_step, spine_angle, ETG_PATH)
    env = GymEnv(theMouse)
    obs_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    # Initialize model, algorithm, agent, replay_memory
    model = MujocoModel(obs_dim, action_dim)
    rpm = ReplayMemory(max_size=MEMORY_SIZE,
                       obs_dim=obs_dim,
                       act_dim=action_dim)
    #这个SAC算法不是很好理解,需要花时间进一步了解下
    algorithm = SAC(model,
                    gamma=GAMMA,
                    tau=TAU,
                    alpha=ALPHA,
                    actor_lr=ACTOR_LR,
                    critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm)
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

    act_bound = np.array([ACT_BOUND_NOW, ACT_BOUND_NOW] * 4)
    # ES_solver.reset()
    if not EVAL:
        #输出配置
        outdir = "./train_log/exp{}".format(EXP_ID)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logger.set_dir(outdir)
        logger.info("slope no mass")
        for ei in range(ES_TRAIN_STEPS):
            start = time.time()
            solutions = ES_solver.ask()
            fitness_list = []
            steps = []
            for id, solution in enumerate(solutions):
                points_add = solution.reshape(-1, 2)
                new_points = prior_points + points_add
                w, b = theController.getETGinfo(new_points)
                theController.update(w, b)
                episode_reward, step = run_EStrain_episode(
                    theMouse, theController, env)
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
                    theMouse, theController, env)
                logger.info('Evaluation Reward: {} step: {} '.format(
                    episode_reward, step))
                summary.add_scalar('EVAL/episode_reward', episode_reward,
                                   ei + 1)
                utility.saveETGinfo(path, w_best, b_best, new_points)
                utility.ETG_trj_plot(w_best, b_best, theController.ETG_agent,
                                     ei, outdir)

    elif EVAL == True:
        print("start eval")
        idx = 180
        ETG_Evalpath = os.path.join(
            script_directory,
            "data/exp6_ETG_models/slopeBest_{}.npz".format(idx))
        info = np.load(ETG_Evalpath)
        w = info["w"]
        b = info["b"]
        print("w.shape:", w.shape)
        print("b.shape:", b.shape)
        # points = info["param"]
        theController.update(w, b)
        utility.ETG_trj_plot(w, b, theController.ETG_agent, idx)
        episode_reward, step = run_EStrain_episode(theMouse, theController,
                                                   env)
        logger.info('Evaluation Reward: {} step: {} '.format(
            episode_reward, step))

    # 安全关闭模拟器
    if theMouse.render:
        theMouse.viewer.close()
    # utility.infoRecord(theMouse, theController)
