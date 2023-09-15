import parl
from locomotionEnv import GymEnv
import numpy as np
from parl.utils import logger, summary
import os
from ToSim import SimModel
from Controller import MouseController
from alg.ETG_alg import SimpleGA
import utility
import time
#_________________________________
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本文件所在的目录
script_directory = os.path.dirname(script_path)
project_path = "/".join(script_directory.split("/")[:-2])

ETG_PATH = os.path.join(script_directory, 'data/ETG_models/Slope_ETG.npz')
ROBOT_PATH = os.path.join(project_path, "ForSim/New/models/dynamic_4l.xml")

RUN_TIME_LENGTH = 8
SIGMA = 0.02
SIGMA_DECAY = 0.99
POP_SIZE = 40
ES_TRAIN_STEPS = 200
EVAL = False
EXP_ID=6
K=2
THREAD=20

@parl.remote_class(wait=False)
class RemoteESAgent(object):
    def __init__(self, time_step,fre_frame,render,fre,spine_angle,prior_points,id):
        self.id = id
        # logger.info("robot_path:{}".format(ROBOT_PATH))
        self.theMouse=SimModel("/mnt/S58Data1/niujh/ForCode/ForSim/New/models/dynamic_4l.xml", time_step, fre_frame, render=render)
        self.theController=MouseController(fre, time_step, spine_angle, ETG_PATH)
        self.env = GymEnv(self.theMouse)
        self.prior_points=prior_points

    def run_EStrain_episode(self):
        obs, info = self.theMouse.reset()
        curFoot = info["curFoot"][1]
        endFoot = 0
        startFoot = curFoot
        terminated = False
        step = 0
        while not terminated:
            tCtrlData = self.theController.runStep()  # No Spine
            step += 1
            #tCtrlData = theController.runStep_spine()		# With Spine
            ctrlData = tCtrlData
            obs, reward, terminated, _, info = self.env.step(ctrlData)
            if step%100==0:
                #防止小鼠偏移既定方向过远
                if(abs(info['curFoot'][0])>0.2):
                    terminated=True
            if step % 1000 == 0:
                endFoot = info["curFoot"][1]
                # print("endFoot=", endFoot)
                dist = endFoot - curFoot
                angle_z = info["euler_z"]
                slope_y=info["slope_y"]
                # print("the move distance of 1000 step:", dist)
                # print("the euler of z axis:", angle_z)
                # print("rot_mat:", info["rot_mat"])
                # time.sleep(1)
                # logger.info("move of x:{}".format(info['curFoot'][0]))
                if (abs(dist) < 5e-4 or dist >= 0.01 or abs(angle_z) > 0.3 ):
                    terminated = True
                if(abs(endFoot-startFoot)>0.5):
                    logger.info("the y pos of slope:{},endFoot_x:{},endFoot_y:{},endFoot_z:{}".format(slope_y,info["curFoot"][0],endFoot,info["curFoot"][2]))
                curFoot = endFoot
        episode_reward = abs(endFoot - startFoot)
        return episode_reward, step

    def sample_episode(self,param):
        points_add = param.reshape(-1, 2)
        new_points = self.prior_points + points_add
        w, b = self.theController.getETGinfo(new_points)
        self.theController.update(w, b)
        episode_reward, step = self.run_EStrain_episode()
        param_ETG={"w":w,"b":b,"points":new_points}
        return episode_reward,step,param_ETG

    def batch_sample_episodes(self, param=None, K=1):
        returns = []
        for i in range(K):
            reward,step,_=self.sample_episode(param[i])
            returns.append((reward, step,self.id * K + i))
        return returns


class ES_ParallelModel():
    def __init__(self,
                 time_step,fre_frame,render,fre,spine_angle,prior_points,
                 num_params=48,
                 K=K,
                 thread=THREAD,
                 sigma=SIGMA,
                 sigma_decay=SIGMA_DECAY,
                 outdir="Results",
                 xparl_addr="localhost:6111"):

        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.num_params = num_params
        self.outdir = outdir
        self.K = K
        self.thread = thread
        self.popsize = K * thread
        self.theController=MouseController(fre, time_step, spine_angle, ETG_PATH)
        print("numparams:", self.num_params)
        print("addr:{}".format(xparl_addr))
        parl.connect(xparl_addr)
        self.agent_list = [
            RemoteESAgent(time_step,fre_frame,render,fre,spine_angle,prior_points,id=i)
            for i in range(self.thread)
        ]
        self.solver = SimpleGA(self.num_params,
                                   sigma_init=self.sigma,
                                   sigma_decay=self.sigma_decay,
                                   sigma_limit=0.02,
                                   elite_ratio=0.1,
                                   weight_decay=0.005,
                                   popsize=self.popsize,
                                   )

    def save(self, epoch,out_dir,param_ETG):
        path=os.path.join(
            script_directory, "data/exp{}_ETG_models".format(EXP_ID))
        if not os.path.exists(path):
            os.makedirs(path)
        path=os.path.join(path,"slopeBest_{}.npz".format(epoch))
        w_best=param_ETG["w"]
        b_best=param_ETG["b"]
        new_points=param_ETG["points"]
        utility.saveETGinfo(path, w_best, b_best, new_points)
        utility.ETG_trj_plot(w_best, b_best, self.theController.ETG_agent,
                                epoch, out_dir)

    def update(self, epoch):
        rewards = []
        solutions = self.solver.ask()
        fitness_list = np.zeros(self.solver.popsize)
        steps=[]
        future_objects = []
        for i in range(self.thread):
            future_objects.append(self.agent_list[i].batch_sample_episodes(
                param=solutions[i * self.K:(i + 1) * self.K, :], K=self.K))
        results_list = [future_obj.get() for future_obj in future_objects]
        for i in range(self.thread):
            results = results_list[i]
            for j in range(self.K):
                result = results[j]
                rewards.append(result[0])
                steps.append(result[1])
                fitness_list[i * self.K + j] = result[0]
        self.solver.tell(fitness_list)
        results = self.solver.result()
        sig = np.mean(results[3])
        rewards = np.asarray(rewards).reshape(-1)
        logger.info('ESSteps: {} Reward: {} step: {}  sigma:{}'.format(
            epoch + 1, np.max(rewards), np.mean(steps), sig))
        summary.add_scalar('ES/episode_reward', np.mean(rewards),
                            epoch + 1)
        summary.add_scalar('ES/episode_minre', np.min(rewards),
                            epoch + 1)
        summary.add_scalar('ES/episode_maxre', np.max(rewards),
                            epoch + 1)
        summary.add_scalar('ES/episode_restd', np.std(rewards),
                            epoch + 1)
        summary.add_scalar('ES/episode_length', np.mean(steps), epoch + 1)
        summary.add_scalar('ES/sigma', sig, epoch + 1)
        return result[1]

    def train(self, epochs=ES_TRAIN_STEPS):

        for epoch in range(epochs):
            start=time.time()
            mean_re = self.update(epoch)
            end=time.time()
            logger.info("time:{}".format(end-start))
            if epoch % 20 == 0 and epoch > 0:
                r,param_ETG = self.evaluate_episode(epoch)
                self.save(epoch, self.outdir,param_ETG)

    def evaluate_episode(self, epoch):
        best_param = self.solver.get_best_param()
        results= self.agent_list[0].sample_episode(param=best_param)
        episode_reward,step,param_ETG =results.get()
        logger.info('Evaluation Reward: {} step: {} '.format(
            episode_reward, step))
        summary.add_scalar('EVAL/episode_reward', episode_reward,
                            epoch + 1)

        return episode_reward,param_ETG

if __name__ == '__main__':

    # logger.info('args:{}'.format(args))
    #_______
    render = False  #控制是否进行画面渲染
    fre_frame = 5  #画面帧率控制或者说小鼠运动速度控制
    fre = 0.5
    time_step = 0.002
    spine_angle = 0  #20
    run_steps_num = int(RUN_TIME_LENGTH / time_step)

    info = np.load(ETG_PATH)
    prior_points = info["param"]
    ETG_param_init = prior_points.reshape(-1)
    outdir = "./train_log/exp{}".format(EXP_ID)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model = ES_ParallelModel(time_step,fre_frame,render,fre,spine_angle,
                             prior_points,num_params=ETG_param_init.shape[0],outdir=outdir)
    if not EVAL:
        logger.set_dir(outdir)
        logger.info("slope no mass")
        model.train()
    elif EVAL == True:
        model.evaluate_episode(0)
    