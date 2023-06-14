"""The swing leg controller class."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import copy
import bisect
import math
import numpy as np
from typing import Any, Mapping, Sequence, Tuple
from mpc_controller import my_swing_example
from mpc_controller import leg_controller

def gen_parabola(phase: float, start: float, mid: float, end: float) -> float:
  """Gets a point on a parabola y = a x^2 + b x + c.

  The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
  the plane.

  Args:
    phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
    start: The y value at x == 0.
    mid: The y value at x == 0.5.
    end: The y value at x == 1.

  Returns:
    The y value at x == phase.
  """

  mid_phase = 0.5
  delta_1 = mid - start
  delta_2 = end - start
  delta_3 = mid_phase**2 - mid_phase
  coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
  coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
  coef_c = start

  return coef_a * phase**2 + coef_b * phase + coef_c

def gen_phase_foot_path_trajectory(input_phase: float, start_pos: Sequence[float], end_pos: Sequence[float], max_clearance = 0.1):
  """Generates the swing trajectory using a parabola.

  Args:
    input_phase: the swing/stance phase value between [0, 1].
    start_pos: The foot's position at the beginning of swing cycle.
    end_pos: The foot's desired position at the end of swing cycle.

  Returns:
    The desired foot position at the current phase.
  """

  phase = input_phase
  # if input_phase <= 0.5:
  #   phase = 0.8 * math.sin(input_phase * math.pi)
  # else:
  #   phase = 0.8 + (input_phase - 0.5) * 0.4

  x = (1 - phase) * start_pos[0] + phase * end_pos[0]
  y = (1 - phase) * start_pos[1] + phase * end_pos[1]
  mid = max(end_pos[2], start_pos[2]) + max_clearance
  z = gen_parabola(phase, start_pos[2], mid, end_pos[2])

  # PyType detects the wrong return type here.
  return (x, y, z)  # pytype: disable=bad-return-type

def select_points(a, n):
  """Selects n points from the input array a."""

  if n <= 0 or n > len(a):
      raise ValueError('n must be between 1 and the length of the input array')
  
  step = (len(a) - 1)//(n - 1)

  indices = np.arange(0, len(a), step, dtype=int)
  if len(indices) < n:
      indices = np.append(indices, len(a)-1)
  indices = indices[:n]
  
  return a[indices]

def find_insertion_index(arr, num):
  return bisect.bisect_left(arr, num)

def minimum_jerk_traj_gen(piece_num,
                          init_pos, init_vel, init_acc,
                          target_pos, target_vel, target_acc,
                          intermediate_positions,
                          control_time_vector):
  # Allocate the M, b matrices with zeros.
  M = np.zeros((piece_num*6, piece_num*6))
  b = np.zeros((piece_num*6, 3))

  # Set the initial conditions.
  F_0 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 2]])
  # print(F_0)
  M[:3, :3] = F_0

  # Stacking initial position, velocity, and acceleration.
  D_0 = np.vstack((init_pos, init_vel, init_acc)).reshape(3,3)
  b[:3, :] = D_0

  # Set the endpoint conditions.
  t_m = control_time_vector[piece_num-1]
  E_M = np.array([[1,      t_m,    t_m**2,    t_m**3,    t_m**4,    t_m**5],
                  [0,        1,     2*t_m,  3*t_m**2,  4*t_m**3,  5*t_m**4],
                  [0,        0,         2,     6*t_m, 12*t_m**2, 20*t_m**3]])
  M[piece_num*6-3:piece_num*6, piece_num*6-6:piece_num*6] = E_M

  # Stacking target position, velocity, and acceleration.
  D_M = np.vstack((target_pos, target_vel, target_acc)).reshape(3,3)
  b[piece_num*6-3:piece_num*6, :] = D_M

  # Set the intermediate conditions.
  for i in range(1, piece_num):
    t_i = control_time_vector[i-1]
    E_i = np.array([[1,      t_i,    t_i**2,    t_i**3,    t_i**4,    t_i**5],
                    [1,      t_i,    t_i**2,    t_i**3,    t_i**4,    t_i**5],
                    [0,        1,     2*t_i,  3*t_i**2,  4*t_i**3,  5*t_i**4],
                    [0,        0,         2,     6*t_i, 12*t_i**2, 20*t_i**3],
                    [0,        0,         0,         6,    24*t_i, 60*t_i**2],
                    [0,        0,         0,         0,        24,   120*t_i]])

    F_i = np.array([[ 0,   0,   0,   0,   0,   0],
                    [-1,   0,   0,   0,   0,   0],
                    [ 0,  -1,   0,   0,   0,   0],
                    [ 0,   0,  -2,   0,   0,   0],
                    [ 0,   0,   0,  -6,   0,   0],
                    [ 0,   0,   0,   0, -24,   0]])

    # Fetch the intermediate positions.
    D_i = intermediate_positions[i-1,:].reshape(1,3)
    # print("D_i is: ", D_i)
    M[6*i-3:6*i+3, (i-1)*6:i*6] = E_i
    M[6*i-3:6*i+3, 6*i:6*i+6] = F_i
    b[6*i-3:6*i-2, :] = D_i

  coefficient_matrix = np.linalg.inv(M)@b

  return coefficient_matrix

def optimizing_foot_path(all_points, time_allocation_vector, control_points_num, take_points_num):
    """Relocate points

    Args:
      all_points: all points include the initial and target points
      time_allocation_vector: time allocation vector includes the initial and target time
      control_points_num: the number of control points used, including the initial and target points
      take_points_num: the number of points to take, including the initial and target points
    """

    if control_points_num < 3:
        raise ValueError("control_points_num should be equal or greater than 3")
    
    if take_points_num < 3:
        raise ValueError("take_points_num should be equal or greater than 3")
    
    control_points = select_points(all_points, control_points_num)
    control_time_vector = select_points(time_allocation_vector, control_points_num)
    control_time_interval = control_time_vector[1] - control_time_vector[0]

    coefficient_matrix = minimum_jerk_traj_gen(piece_num=control_points_num - 1,
                                               init_pos=control_points[0,:],
                                               init_vel=np.array([0, 0, 0]),
                                               init_acc=np.array([0, 0, 0]),
                                               target_pos=control_points[-1,:],
                                               target_vel=np.array([0, 0, 0]),
                                               target_acc=np.array([0, 0, 0]),
                                               intermediate_positions=control_points[1:-1,:],
                                               control_time_vector=np.ones(control_points_num)*control_time_interval)
    print("coefficient_matrix is: ", coefficient_matrix)
    opt_points = np.zeros((take_points_num, 3))
    
    for i in range(take_points_num):
      t_abs = control_time_vector[-1]/take_points_num*i
      j = find_insertion_index(control_time_vector, t_abs)

      # j = int(t_abs/control_time_interval)
      if j == 0:
        j = 1
      t = t_abs%control_time_interval
      t_array = np.array([[1, t, t**2, t**3, t**4, t**5]]).reshape(1,6)

      opt_points[i,0] = t_array @ coefficient_matrix[j*6-6:j*6,0]
      opt_points[i,1] = t_array @ coefficient_matrix[j*6-6:j*6,1]
      opt_points[i,2] = t_array @ coefficient_matrix[j*6-6:j*6,2]

    # correction
    for i in range(control_points_num-1):
      opt_points[i*int(take_points_num/(control_points_num-1)), :] = opt_points[i*int(take_points_num/(control_points_num-1)) - 1, :]
    opt_points[-1,:] = all_points[-1,:]

    return opt_points

def collision_check(foot_pos, leg_size, obstacle_pos, obstacle_size):
  """Check the collision between the leg and the obstacle in XZ plane."""
  
  checkpoint_1_XZ = np.array([obstacle_pos[0] - obstacle_size[0], 
                              obstacle_pos[2] + obstacle_size[2]])
  checkpoint_2_XZ = np.array([obstacle_pos[0] + obstacle_size[0], 
                              obstacle_pos[2] + obstacle_size[2]])
  foot_pos_XZ = np.array([foot_pos[0], foot_pos[2]])
  k = 0.5
  if np.linalg.norm(foot_pos_XZ - checkpoint_1_XZ) <= leg_size + k * obstacle_size[2]:
      # print("Collision detected!")
      return True
  elif np.linalg.norm(foot_pos_XZ - checkpoint_2_XZ) <= leg_size + k * obstacle_size[2]:
      # print("Collision detected!")
      return True
  return False

class MySwingLegController(leg_controller.LegController):
  """Controls the swing leg position using Raibert's formula."""

  def __init__(self,
               robot: Any,
               desired_speed: np.ndarray,
               desired_twisting_speed: float,
               desired_height: float):
    """Initializes the class.

    Args:
      robot: A robot instance.
      desired_speed: Behavior parameters. X-Y speed.
      desired_twisting_speed: Behavior control parameters.
      desired_height: Desired standing height.
      foot_clearance: The foot clearance on the ground at the end of the swing
        cycle.
    """
    self._robot = robot
    self._desired_speed = desired_speed
    self._desired_twisting_speed = desired_twisting_speed
    self._desired_height = desired_height
    self._joint_angles = None
    self._phase_switch_foot_local_position = None
    self.reset(0)

  def reset(self, current_time: float) -> None:
    """Called during the start of a swing cycle.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    self._phase_switch_foot_local_position = (
        self._robot.GetFootPositionsInBaseFrame())
    self._joint_angles = {}

  def _get_foot_path(self,
                     foot_init_positions,
                     foot_target_positions,
                     isSingleFRLeg=True,
                     max_clearance=0.05,
                     phase_num=500):
    """Get the foot path trajectory."""

    foot_path = np.zeros((phase_num, 4, 3))

    for i in range(phase_num):
      phase = i / phase_num
      if isSingleFRLeg:
        foot_path[i][0] = gen_phase_foot_path_trajectory(phase, foot_init_positions[0], foot_target_positions[0], max_clearance)
        foot_path[i][1] = foot_init_positions[1]
        foot_path[i][2] = foot_init_positions[2]
        foot_path[i][3] = foot_init_positions[3]
      else:
        for leg_id in range(4):
          foot_path[i][leg_id] = gen_phase_foot_path_trajectory(phase, foot_init_positions[leg_id], foot_target_positions[leg_id], max_clearance)
    
    return foot_path

  def get_foot_path(self,
                    foot_init_positions,
                    foot_target_positions,
                    obstacle_pos,
                    obstacle_size,
                    max_clearance=0.05,
                    phase_num=500,
                    isSingleFRLeg=True,
                    withObstacle=False,
                    withOptimization=True,
                    save_path='',
                    savetxt_parameters=''):
    """Get the foot path from initial position to target position."""

    if len(foot_init_positions) != 4 or len(foot_target_positions) != 4:
      raise ValueError("The length of foot_init_positions and foot_target_positions must be 4, even if isSingleLeg is True.")
    
    if len(foot_init_positions[0]) != 3 or len(foot_target_positions[0]) != 3:
      raise ValueError("The length of foot_init_positions[0] and foot_target_positions[0] must be 3.")
    
    if len(obstacle_pos) != 3 or len(obstacle_size) != 3:
      raise ValueError("The length of obstacle_pos and obstacle_size must be 3.")

    control_points_num = 3

    if not withObstacle:
      foot_path = self._get_foot_path(foot_init_positions, foot_target_positions, isSingleFRLeg, max_clearance=max_clearance, phase_num=phase_num)
      with open("%sfoot_path%s.txt"%(save_path, savetxt_parameters), 'w') as f:
        np.savetxt(f, foot_path[:, 0, :], delimiter=',')
      if withOptimization:
        foot_path[:, 0, :] = optimizing_foot_path(all_points=foot_path[:, 0, :],
                                                  time_allocation_vector=np.linspace(0, 1, phase_num),
                                                  control_points_num=control_points_num,
                                                  take_points_num=phase_num)
        with open("%sfoot_opt_path%s.txt"%(save_path, savetxt_parameters), 'w') as f:
          np.savetxt(f, foot_path[:, 0, :], delimiter=',')
        return foot_path
      else:
        return foot_path
    
    isCollision = True
    leg_size = 0.03
    while isCollision:
      foot_path = self._get_foot_path(foot_init_positions, foot_target_positions, isSingleFRLeg, max_clearance=max_clearance, phase_num=phase_num)
      collision = False
      for foot_pos in foot_path[:, 0, :]:
        foot_pos_in_world_frame = self.foot_pos_in_world_frame_from_local_frame(foot_pos)
        if collision_check(foot_pos_in_world_frame,
                           leg_size,
                           obstacle_pos,
                           obstacle_size):
          collision = True
          max_clearance += 0.01
          break
      if not collision:
        isCollision = False
    with open("%sfoot_path%s.txt"%(save_path, savetxt_parameters), 'w') as f:
      np.savetxt(f, foot_path[:, 0, :], delimiter=',')
    if withOptimization:
      foot_path[:, 0, :] = optimizing_foot_path(all_points=foot_path[:, 0, :],
                                                time_allocation_vector=np.linspace(0, 1, phase_num, endpoint=False) + 1/phase_num,
                                                control_points_num=control_points_num,
                                                take_points_num=phase_num)
      with open("%sfoot_opt_path%s.txt"%(save_path, savetxt_parameters), 'w') as f:
        np.savetxt(f, foot_path[:, 0, :], delimiter=',')
      return foot_path
    
    return foot_path
  
  def foot_pos_in_world_frame_from_local_frame(self, local_frame_position):
    """Converts a local frame position to a world frame position.

    Args:
      local_frame_position: A 3D vector in the local frame.

    Returns:
      A 3D vector in the world frame.
    """
    foot_size = 0.02
    local_frame = np.array(local_frame_position).reshape(3)
    world_frame = local_frame + np.array([0, 0, my_swing_example._ROBOT_BASE_HEIGHT - foot_size])
    return world_frame

  def get_action(self, foot_target_positions):
    _joint_angles = {}
    for leg_id in range(4):
      _, _, yaw_dot = self._robot.GetBaseRollPitchYawRate()
      hip_positions = self._robot.GetHipPositionsInBaseFrame()
      hip_offset = hip_positions[leg_id]
      twisting_vector = np.array((-hip_offset[1], hip_offset[0], 0))
      joint_ids, joint_angles = (
          self._robot.ComputeMotorAnglesFromFootLocalPosition(
              leg_id, foot_target_positions[leg_id]))
      # Update the stored joint angles as needed.
      for joint_id, joint_angle in zip(joint_ids, joint_angles):
        _joint_angles[joint_id] = (joint_angle, leg_id)

    action = {}
    kps = self._robot.GetMotorPositionGains()
    kds = self._robot.GetMotorVelocityGains()
    for joint_id, joint_angle_leg_id in _joint_angles.items():
      leg_id = joint_angle_leg_id[1]
      action[joint_id] = (joint_angle_leg_id[0], kps[joint_id], 0,
                            kds[joint_id], 0)
    
    actionList = []
    for joint_id in range(self._robot.num_motors):
          actionList.extend(action[joint_id])
    actionList = np.array(actionList, dtype=np.float32)
    
    return actionList
