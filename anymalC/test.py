import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('scene.xml')
d = mujoco.MjData(m)

viewer = mujoco.viewer.launch_passive(m, d)

while viewer.is_running():
    step_start = time.time()
    # print(d)
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    # with viewer.lock():
    #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time %
    #                                                                  2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    #这里应该等价于mujoco_py的viewer.render()函数,用于获取最新的模型状态
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    # time_until_next_step = m.opt.timestep - (time.time() - step_start)
    # if time_until_next_step > 0:
    #     time.sleep(time_until_next_step)
print("pass")
# with mujoco.viewer.launch_passive(m, d) as viewer:
#     # Close the viewer automatically after 30 wall-seconds.
#     start = time.time()
#     while viewer.is_running() and time.time() - start < 30:
#         step_start = time.time()
#         # print(d)
#         # mj_step can be replaced with code that also evaluates
#         # a policy and applies a control signal before stepping the physics.
#         mujoco.mj_step(m, d)

#         # Example modification of a viewer option: toggle contact points every two seconds.
#         with viewer.lock():
#             viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
#                 d.time % 2)

#         # Pick up changes to the physics state, apply perturbations, update options from GUI.
#         #这里应该等价于mujoco_py的viewer.render()函数,用于获取最新的模型状态
#         viewer.sync()

#         # Rudimentary time keeping, will drift relative to wall clock.
#         time_until_next_step = m.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)
