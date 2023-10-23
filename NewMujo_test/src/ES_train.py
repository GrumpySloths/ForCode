from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
from locomotionRLEnv import GymEnv_RL,Param_Dict,VEL_D_BODY,VEL_D_FOOT
import time
import numpy as np
import utility
from alg.ETG_alg import SimpleGA
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append("..")
from parl.utils import logger, summary,ReplayMemory
from model.mujoco_agent import MujocoAgent
from model.mujoco_model import MujocoModel
from alg.sac import SAC
from copy import copy
import torch
import argparse
# --------------------
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本文件所在的目录
script_directory = os.path.dirname(script_path)
project_path = "/".join(script_directory.split("/")[:-2])
print("hello world")
RL_TRAIN=True  #是否进行ETG_RL训练
ES=False #是否中途进行ETG的训练
DEBUG = False #用于debug打印信息
ETG_PATH = os.path.join(script_directory, 'data/success_models/slopeBest_60.npz')
ROBOT_PATH = os.path.join(project_path, "ForSim/New/models/dynamic_4l.xml")
RUN_TIME_LENGTH = 8
SIGMA = 0.02
SIGMA_DECAY = 0.99
POP_SIZE = 40
ES_TRAIN_STEPS = 200
EVAL = False
EXP_ID = 7
#___________RL_ETG配置_____________
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ACT_BOUND_NOW=0.3
MEMORY_SIZE = int(1e6)
WARMUP_STEPS = 1e4
BATCH_SIZE = 256
MAX_STEPS=1e7
EPISODE_MAXSTEP=400
EVAL_EVERY_STEPS = 1e4
ES_EVERY_STEPS = 5e4
ETG_TRAIN_STPES=30
ETG_TRAIN_STPES_LOWBOUND=10 #ETG训练步骤的下限
#______________________________
def debug(info):
    if DEBUG:
        print(info)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=int, default=0, help="Evaluate or Not")
    parser.add_argument("--debug", type=int, default=0, help="debug or Not")
    parser.add_argument("--eval_ModelID", type=int, default=0, help="Evaluate时选取的模型序号")
    parser.add_argument("--exp_id", type=int, default=11, help="用于设置输出文件的id号")
    parser.add_argument("--foot_gain", type=float, default=1.0, help="用于设置Reward中foot所占增益比例")
    parser.add_argument("--body_gain", type=float, default=0.2, help="用于设置Reward中body所占增益比例")
    
    return parser
    
def run_ES_RL_train_episode(agent,env,rpm,max_step,action_bound,w=None,b=None):

    obs, info = env.reset(ETG_w=w,ETG_b=b)
    episode_reward, episode_steps = 0, 0
    critic_loss_list = []
    actor_loss_list = []
    infos={}
    success_num=0
    terminated = False
    while not terminated:
        episode_steps+=1
        # Select action randomly or according to policy
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)
        new_action = copy(action)
        # Perform action
        next_obs, reward, terminated, _, info = env.step(new_action*action_bound)
        # Store data in replay memory
        terminal=1.-float(terminated)
        rpm.append(obs, action, reward, next_obs, terminal)
        obs = next_obs
        episode_reward += reward
        #对Reward的各部分添加探针
        for key in Param_Dict.keys():
            if key in info.keys():
                if key not in infos.keys():
                    infos[key] = info[key]
                else:
                    infos[key] += info[key]
        if info['vel_body']>VEL_D_BODY and abs(info['vel_foot']-VEL_D_FOOT)<0.025:
            success_num+=1
        # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            # print("开始学习")
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            critic_loss, actor_loss = agent.learn(batch_obs, batch_action,
                                                  batch_reward, batch_next_obs,
                                                  batch_terminal)
            critic_loss_list.append(critic_loss)
            actor_loss_list.append(actor_loss)
        if episode_steps > max_step:
            break
    if len(critic_loss_list) > 0:
        infos["critic_loss"] = np.mean(critic_loss_list)
        infos["actor_loss"] = np.mean(actor_loss_list)
    infos["success_rate"] = success_num / episode_steps

    return episode_reward, episode_steps,infos


def run_Evaluate_episode(agent,env,max_step,action_bound,w=None,b=None):
    obs, info = env.reset(ETG_w=w,ETG_b=b)
    episode_reward, episode_steps = 0, 0
    infos={}
    terminated=False
    while not terminated:
        episode_steps+=1
        action = agent.predict(obs)
        # Perform action
        next_obs, reward, terminated, _, info = env.step(action*action_bound)
        obs = next_obs
        episode_reward += reward
        if episode_steps > max_step:
            break

    return episode_reward, episode_steps,infos



if __name__ == '__main__':
    parser=make_parser()
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        logger.info("cuda is avaiable")
    #_______
    render = False  #控制是否进行画面渲染
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
    #___________RL_ETG配置_____________
    act_bound = np.array([ACT_BOUND_NOW, ACT_BOUND_NOW] * 4)
    # Initialize model, algorithm, agent, replay_memory
    env=GymEnv_RL(theMouse,theController,debug_stat=args.debug,foot_gain=args.foot_gain,body_gain=args.body_gain)
    obs_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
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

    if not args.eval :
        EvalReward_Max=0  #只保留结果最好的几个模型
        eval_avg_reward=0 #存储当前eval Reward
        cur_ETG_train_steps=ETG_TRAIN_STPES
        #输出配置
        outdir = "./train_log/exp{}".format(args.exp_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logger.set_dir(outdir)
        logger.info("ETG_RL train start:")
        if ES:
            logger.info("ETG train Included")
        logger.info('args:{}'.format(args))
        #_______________训练过程________________
        total_steps = 0
        test_flag = 0
        ES_test_flag = 0
        ES_steps=0
        # t_steps = 0
        #ETG info init
        best_param = ES_solver.get_best_param()
        points_add = best_param.reshape(-1, 2)
        new_points = prior_points + points_add
        w_best, b_best = theController.getETGinfo(new_points)

        while total_steps<MAX_STEPS:
            episode_reward, episode_step, info = run_ES_RL_train_episode(
                agent, env, rpm, EPISODE_MAXSTEP, act_bound, w_best, b_best)
            
            total_steps+=episode_step
            summary.add_scalar('train/episode_reward', episode_reward,
                               total_steps)
            summary.add_scalar('train/episode_step', episode_step, total_steps)
            for key in info.keys():
                if info[key] != 0:
                    summary.add_scalar('train/episode_{}'.format(key),
                                       info[key], total_steps)
                    summary.add_scalar('train/mean_{}'.format(key),
                                       info[key] / episode_step, total_steps)
            logger.info('Total Steps: {} Reward: {}'.format(
                total_steps, episode_reward))
            # ————————————————Evaluate episode——————————————————
            if (total_steps + 1) // EVAL_EVERY_STEPS > test_flag:
                while (total_steps + 1) // EVAL_EVERY_STEPS > test_flag:
                    test_flag += 1
                    eval_avg_reward, avg_step, info = run_Evaluate_episode(
                        agent, env, 600, act_bound, w_best, b_best)
                    logger.info(
                        'Evaluation Process, Reward: {} Steps: {} '.
                        format(eval_avg_reward, avg_step))
                    summary.add_scalar('eval/episode_reward', eval_avg_reward,
                                       total_steps)
                    summary.add_scalar('eval/episode_step', avg_step,
                                       total_steps)

                path = os.path.join(script_directory,
                                    "data/exp{}_ETG_RL_models".format(args.exp_id))
                if not os.path.exists(path):
                    os.makedirs(path)
                if eval_avg_reward>EvalReward_Max:
                    EvalReward_Max=eval_avg_reward
                    path_rl = os.path.join(path, "itr_{}.pt".format(total_steps))
                    path_ETG = os.path.join(path, "itr_{}.npz".format(total_steps))
                    agent.save(path_rl)
                    utility.saveETGinfo(path_ETG,
                            w_best, b_best, new_points)
                    utility.updateFolder(model_folder=path,num_models_to_keep=6)
            #_______________ETG evolution Process__________
            if ES and (total_steps + 1) // ES_EVERY_STEPS >= ES_test_flag and total_steps >= WARMUP_STEPS:
                while (total_steps + 1) // ES_EVERY_STEPS >= ES_test_flag:
                    ES_test_flag += 1

                    for ei in range(cur_ETG_train_steps):
                        ES_steps+=1
                        start = time.time()
                        solutions = ES_solver.ask()
                        fitness_list = []
                        steps = []
                        for id, solution in enumerate(solutions):
                            points_add = solution.reshape(-1, 2)
                            new_points = prior_points + points_add
                            w, b = theController.getETGinfo(new_points)
                            episode_reward, step,info = run_Evaluate_episode(
                                agent, env, 600, act_bound, w_best, b_best)
                            steps.append(step)
                            fitness_list.append(episode_reward)
                        results = ES_solver.result()
                        sig = np.mean(results[3])
                        fitness_list = np.asarray(fitness_list).reshape(-1)
                        ES_solver.tell(fitness_list)
                        end = time.time()
                        logger.info(
                            'ESSteps: {} Reward: {} step: {}  sigma:{},time:{}'.format(
                                ES_steps, np.max(fitness_list), np.mean(steps), sig,
                                end - start))
                        summary.add_scalar('ES/episode_reward', np.mean(fitness_list),
                                        ES_steps)
                        summary.add_scalar('ES/episode_minre', np.min(fitness_list),
                                        ES_steps)
                        summary.add_scalar('ES/episode_maxre', np.max(fitness_list),
                                        ES_steps)
                        summary.add_scalar('ES/episode_restd', np.std(fitness_list),
                                        ES_steps)
                        summary.add_scalar('ES/episode_length', np.mean(steps), ES_steps)
                        summary.add_scalar('ES/sigma', sig, ES_steps)

                cur_ETG_train_steps=int(cur_ETG_train_steps*0.9)
                if cur_ETG_train_steps<ETG_TRAIN_STPES_LOWBOUND:
                    cur_ETG_train_steps=ETG_TRAIN_STPES_LOWBOUND
                ETG_best_param = ES_solver.get_best_param()
                points_add = ETG_best_param.reshape(-1, 2)
                new_points = prior_points + points_add
                w_best, b_best = theController.getETGinfo(new_points)
                ES_solver.reset(ETG_best_param)

    if  args.eval ==True:
        logger.info("ES_rl eval start:")
        id=args.eval_ModelID
        path = os.path.join(script_directory,
                            "data/exp{}_ETG_RL_models".format(args.exp_id))
        path_rl = os.path.join(path, "itr_{}.pt".format(id))
        path_ETG = os.path.join(path, "itr_{}.npz".format(id))
        agent.restore(path_rl)
        info = np.load(path_ETG)
        w = info["w"]
        b = info["b"]
        avg_reward, avg_step, info = run_Evaluate_episode(agent, env, 600, act_bound, w, b)
        logger.info(
            'Evaluation Process, Reward: {} Steps: {} '.
            format(avg_reward, avg_step))

    # 安全关闭模拟器
    if theMouse.render:
        theMouse.viewer.close()
    # utility.infoRecord(theMouse, theController)
