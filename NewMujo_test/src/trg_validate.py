from LegModel.legs import LegModel
import numpy as np
import matplotlib.pyplot as plt
from Controller import MouseController
import os
import utility
import argparse
ETG_T = 2

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=3, help="用于设置输出文件的id号")
    parser.add_argument("--eval_ModelID", type=int, default=400, help="Evaluate时选取的模型序号")

    return parser

def ETG_trj_plot(ETG_agent,w,b):
    ts_new=np.linspace(0,ETG_T,100)
    obs1=[]
    for t in ts_new:
        v=ETG_agent.update(t)
        obs1.append(v)
    obs1=np.asarray(obs1).reshape(-1,20)
    points_new=obs1.dot(w)+b
    plt.plot(points_new[:,0],points_new[:,1],'g-',linewidth=3)

def pointsCheck(points,legmodel):
    '''做一个简单的测试验证points对应的轨迹是否合法'''
    for point in points:
        q1,q2=legmodel.pos_2_angle(*point)
        print("q1={},q2={}".format(q1,q2))
        if q1<-0.5*np.pi or q1>0.5*np.pi:
            return False
        if q2<-0.5*np.pi or q2>0.5*np.pi:
            return False
    return True
# --------------------
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本文件所在的目录
script_directory = os.path.dirname(script_path)

parser=make_parser()
args = parser.parse_args()
ETG_PATH = os.path.join(script_directory, 'data/exp{}_ETG_models/slopeBest_{}.npz'.format(args.exp_id, args.eval_ModelID))

leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]

legmodel=LegModel(leg_params)
result=[]
for q1 in np.linspace(-0.5*np.pi,0.5*np.pi,180):
    for q2 in np.linspace(-0.5*np.pi,0.5*np.pi,180):
        result.append(legmodel.angel_2_pos(q1,q2))
result=np.array(result)

fre_frame = 5  #画面帧率控制或者说小鼠运动速度控制
fre = 0.5
time_step = 0.005
spine_angle = 0  #20
theController = MouseController(fre, time_step, spine_angle, ETG_PATH)
w,b,points_ETG=theController.ETG_model.getModelInfo()
if pointsCheck(points_ETG,legmodel):
    print("轨迹采样点合法")
else:
    print("轨迹采样点不合法")
ETG_trj_plot(theController.ETG_agent,w,b)
plt.scatter(result[:,0],result[:,1])
plt.scatter(points_ETG[:,0],points_ETG[:,1])
plt.savefig("./plot_log/trj_validate_exp{}_id{}".format(args.exp_id,args.eval_ModelID))
# plt.show()



