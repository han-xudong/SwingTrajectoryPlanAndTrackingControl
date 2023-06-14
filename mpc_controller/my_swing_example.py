from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from typing import Any, Mapping, Sequence, Tuple
import os
import math
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from absl import app
# from absl import flags
# import scipy.interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client
import time
import pybullet
from mpc_controller import my_swing_leg_controller
from mpc_controller import my_a1_sim as robot_sim

_RECORD_VIDEO = False #recording video requires ffmpeg in the path
_MAX_TIME_SECONDS = 10
_NUM_BULLET_SOLVER_ITERATIONS = 30
_SIMULATION_TIME_STEP = 0.001
_ROBOT_BASE_HEIGHT = 0.33
_WITH_OBSTACLE = True
_OBSTACLE_SIZE = [0.015, 1, 0.03] # thickness, width, height
_OBSTACLE_POS = [0.175, 0, 0]
_MAX_CLEARANCE = 0.02
_PHASE_NUM = 500
_WITH_OPTIMIZATION = True

def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""

  desired_speed = (0, 0)
  desired_twisting_speed = 0
  controller = my_swing_leg_controller.MySwingLegController(robot,
                                                            desired_speed=desired_speed,
                                                            desired_twisting_speed=desired_twisting_speed,
                                                            desired_height=robot_sim.MPC_BODY_HEIGHT)
  
  return controller

def _run_example(max_time=_MAX_TIME_SECONDS):
  """Runs the locomotion controller example."""

  if _RECORD_VIDEO:
    p = pybullet
    video_name = "my_swing_example" + \
                 "_with_obstacle_of_height_%.2f"%_OBSTACLE_SIZE[2] if _WITH_OBSTACLE else "my_swing_example" + \
                 "_optimized" if _WITH_OPTIMIZATION else "_unoptimized"
    p.connect(p.GUI, options="--width=1280 --height=720 --mp4=\"%s.mp4\" --mp4fps=100"%video_name)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
  else:
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)    
  p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  p.setAdditionalSearchPath(pd.getDataPath())
  p.setPhysicsEngineParameter(numSolverIterations=_NUM_BULLET_SOLVER_ITERATIONS)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setPhysicsEngineParameter(numSolverIterations=30)
  p.setTimeStep(_SIMULATION_TIME_STEP)
  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pd.getDataPath())
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
  p.resetDebugVisualizerCamera(cameraDistance=0.45,
                               cameraYaw=0,
                               cameraPitch=0,
                               cameraTargetPosition=[_OBSTACLE_POS[0],0,0.1])
    
  planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
  ground_id  = p.createMultiBody(0, planeShape)

  robot_base_height = _ROBOT_BASE_HEIGHT
  robot_uid = p.loadURDF(robot_sim.URDF_NAME, [0, 0, robot_base_height])
  robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=_SIMULATION_TIME_STEP)
  cid2 = p.createConstraint(ground_id, -1, robot_uid, 0, p.JOINT_FIXED, [0, 0, 0], [0, 0, robot_base_height], [0, 0, 0])
  
  controller = _setup_controller(robot)
  controller.reset(current_time=robot.GetTimeSinceReset())
  
  foot_size = 0.02
  foot_init_positions = np.zeros((4,3))
  foot_init_positions[0] = [0., -0.16, -robot_base_height + foot_size] #FR
  foot_init_positions[1] = [0., 0.16, -robot_base_height + foot_size] #FL
  foot_init_positions[2] = [-0.18, -0.16, -robot_base_height + foot_size] #RR
  foot_init_positions[3] = [-0.18, 0.16, -robot_base_height + foot_size] #RL

  foot_target_positions = foot_init_positions + np.array([[0.3, 0, 0], #FR
                                                          [0, 0, 0], #FL
                                                          [0, 0, 0], #RR
                                                          [0, 0, 0]]) #RL

  init_time = time.time()
  while time.time() - init_time < 0.1:
    p.stepSimulation()
    init_action = controller.get_action(foot_init_positions)
    robot.Step(init_action)
  
  obstacle_size = _OBSTACLE_SIZE # thickness, width, height
  obstacle_pos = _OBSTACLE_POS
  if _WITH_OBSTACLE:
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          halfExtents=obstacle_size)
    collison_box_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                             halfExtents=obstacle_size)
    obstacle_id = p.createMultiBody(baseMass=10000,
                                    baseCollisionShapeIndex=collison_box_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=[0.175, 0, 0])
    cid1 = p.createConstraint(ground_id, -1, obstacle_id, -1, p.JOINT_FIXED, [0, 0, 0], [obstacle_pos[0], obstacle_pos[1], obstacle_size[2]], [0, 0, 0])

  foot_path = controller.get_foot_path(foot_init_positions=foot_init_positions,
                                       foot_target_positions=foot_target_positions,
                                       obstacle_pos=obstacle_pos,
                                       obstacle_size=obstacle_size,
                                       max_clearance=_MAX_CLEARANCE,
                                       phase_num=_PHASE_NUM,
                                       isSingleFRLeg=True,
                                       withObstacle=_WITH_OBSTACLE,
                                       withOptimization=_WITH_OPTIMIZATION)
  with open('foot_path.txt', 'w') as f:
    np.savetxt(f, foot_path[:, 0, :], delimiter=',')

  init_time = time.time()
  while time.time() - init_time < 1:
    init_action = controller.get_action(foot_init_positions)
    robot.Step(init_action)
  
  foot_current_position = np.array(robot.GetFootPositionsInWorldFrame())
  with open('foot_real_path.txt', 'w') as f:
    np.savetxt(f, foot_current_position[0].reshape(1, 3), delimiter=',')
  foot_real_path_list = [foot_current_position]
  cnt = 0
  current_time = robot.GetTimeSinceReset()
  while current_time < max_time:
    p.stepSimulation()    
    p.submitProfileTiming("loop")

    if cnt < len(foot_path) - 1:
      cnt += 1
    actions = controller.get_action(foot_path[cnt])
    robot.Step(actions)
    
    foot_current_position = np.array(robot.GetFootPositionsInWorldFrame())
    with open('foot_real_path.txt', 'a') as f:
      np.savetxt(f, foot_current_position[0].reshape(1, 3), delimiter=',')
    foot_real_path_list.append(foot_current_position)
    if foot_real_path_list.__len__() > 1:
      p.addUserDebugLine(foot_real_path_list[-2][0],
                         foot_real_path_list[-1][0],
                         lineColorRGB=[1, 0, 0],
                         lifeTime=20,
                         lineWidth=3)

    if _RECORD_VIDEO:
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    
    current_time = robot.GetTimeSinceReset()
    p.submitProfileTiming()
    time.sleep(5/_PHASE_NUM)

def main(argv):
  del argv
  _run_example()

if __name__ == "__main__":
  app.run(main)
