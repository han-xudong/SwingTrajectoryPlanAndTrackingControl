# SwingTrajectoryPlanAndTrackingControl

In this project, we focus on the planning and control for the swing trajectories.

The codes are based on: <https://xbpeng.github.io/projects/Robotic_Imitation/index.html>

## Getting Started

We use this repository with Python 3.7 or Python 3.8 on Ubuntu, MacOS and Windows.

Install dependencies:

- Install MPC extension (Optional) `python3 setup.py install --user`

- Install MPI: `sudo apt install libopenmpi-dev`

- Install requirements: `pip3 install -r requirements.txt`

You can run `mpc_controller/my_swing_example.py` to get a quickstart.

Here are the definitions of the importtant varibles in `mpc_controller/my_swing_example.py`:

- `_RECORD_VIDEO`: `True` or `False` to record a video of the simulation process or not, which requires ffmpeg in the path.

- `_MAX_TIME_SECONDS`: the maximum running time (seconds) of the simulatioin process.

- `_NUM_BULLET_SOLVER_ITERATIONS`: the number of iterations of the bullet solver.

- `_SIMULATION_TIME_STEP`: the time step of the simulation.

- `_ROBOT_BASE_HEIGHT`: the fixed height of the robot base.

- `_WITH_OBSTACLE`: `True` or `False` to show an obstacle or not.

- `_OBSTACLE_HALF_SIZE`: the half size of the obstacle.

- `_OBSTACLE_POS`: the position of the obstacle.

- `_MAX_CLEARANCE`: the maximum clearance of the foot path to the ground.

- `_PHASE_NUM`: the number of the phases of the swing trajectory.

- `_WITH_OPTIMIZATION`: `True` or `False` to use the optimization method or not.

- `_FOOT_STEP_DISP`: the displacement matrix (4x3) of the 4 foot steps, which are in the robot local frame.

- `_PARAMETERS`: collection of the parameters of the swing trajectory to be saved as the name of the result file.

- `_SAVE_PATH`: the path to save the result file.

## Finding a feasible trajectory connecting the initial and final positions

We give a method to find a feasible swing trajectory when given an initial and final position of the foot tip. And we also design a tracking controller to track the planned trajectory. To simplify, we only consider the single FR leg (foot order is 0) and fixed the robot base at a certain height defined by `_ROBOT_BASE_HEIGHT`.

In `mpc_controller/my_swing_example.py`, the initial positions of the 4 foot tips are (0, -0.16, -0.31), (0, 0.16, -0.31), (-0.18, -0.16, -0.31), and (-0.18, 0.16, -0.31) in the robot local frame. The final position is based on `_FOOT_STEP_DISP` which is a 4x3 matrix describing all 4 foot displacements. We can change `_FOOT_STEP_DISP` to give different swing path.

We also provide an optimization method to find a better trajectory. To use the optimization method, `_WITH_OPTIMIZATION` need to be `True`. You can find the modules `optimizing_foot_path` in `mpc_controller/my_swing_leg_controller.py`. The optimization method is not always successful especially too much control points input, but it can find a better trajectory when it is successful.

The results are in the folder `./results`. Different results are collected in different folders in the format of `my_swing_example_disp_` + `_FOOT_STEP_DISP[0][0]` + `_FOOT_STEP_DISP[0][1]` + `_optimized`(optional based on `_WITH_OPTIMIZATION`).

## Finding a collision-free trajectory when there are known obstacles

In the presence of known obstacles, a swing trajectory can also be generated by the proposed method with a collision avoidence module. We tested the trajectory generating method with obtacles of different height of 0.02, 0.04, 0.06, and 0.08 mm. All the tests were successful, which means that the method is avalible.

In `mpc_controller/my_swing_example.py`, to show an obstacle, which is a cuboid, `_WITH_OBSTACLE` need to be `True`. `_OBSTACLE_HALF_SIZE` and `_OBSTACLE_POS` define the size and position of the obstacle.

The module `collision_check` in `mpc_controller/my_swing_leg_controller.py` is to check whether the generated swing path is collision-free or not, and if not, the path is regenerated until it is collision-free.

The results are in the folder `./results`. Different results are collected in different folders in the format of `my_swing_example_disp_` + `_FOOT_STEP_DISP[0][0]` (displacement in x direction) + `_FOOT_STEP_DISP[0][1]` (displacement in y direction) + `_obstacle_` + `_OBSTACLE_HALF_SIZE`*2 (the height of the obstacle) + `_optimized`(optional based on `_WITH_OPTIMIZATION`).

## Results

### Single leg swing trajectory with different foot displacements

The following figures show the swing trajectories with different foot displacements. The displacements are (0.1, -0.1, 0), (0.1, 0, 0), (0.1, 0.1, 0), (0.2, -0.1, 0), (0.2, 0, 0), (0.2, 0.1, 0), (0.3, -0.1, 0), (0.3, 0, 0), and (0.3, 0.1, 0). Here we show the trajectories with and without optimization.

![swing_trajectory_1](./results/my_swing_example_disp_0.10_-0.10_optimized/path_disp_0.10_-0.10_optimized.png)![swing_trajectory_2](./results/my_swing_example_disp_0.10_0.00_optimized/path_disp_0.10_0.00_optimized.png)![swing_trajectory_3](./results/my_swing_example_disp_0.10_0.10_optimized/path_disp_0.10_0.10_optimized.png)

![swing_trajectory_4](./results/my_swing_example_disp_0.20_-0.10_optimized/path_disp_0.20_-0.10_optimized.png)![swing_trajectory_5](./results/my_swing_example_disp_0.20_0.00_optimized/path_disp_0.20_0.00_optimized.png)![swing_trajectory_6](./results/my_swing_example_disp_0.20_0.10_optimized/path_disp_0.20_0.10_optimized.png)

![swing_trajectory_7](./results/my_swing_example_disp_0.30_-0.10_optimized/path_disp_0.30_-0.10_optimized.png)![swing_trajectory_8](./results/my_swing_example_disp_0.30_0.00_optimized/path_disp_0.30_0.00_optimized.png)![swing_trajectory_9](./results/my_swing_example_disp_0.30_0.10_optimized/path_disp_0.30_0.10_optimized.png)

### Single leg swing trajectory with different obstacles

The following figures show the swing trajectories with different obstacles. The obstacles are cuboids with height of 0.02, 0.04, 0.06, and 0.08 mm. Here we show the trajectories with and without optimization.

![swing_trajectory_1](./results/my_swing_example_disp_0.30_0.00_obstacle_0.02/my_swing_example_disp_0.30_0.00_obstacle_0.02.png)![swing_trajectory_1](./results/my_swing_example_disp_0.30_0.00_obstacle_0.04/my_swing_example_disp_0.30_0.00_obstacle_0.04.png)

![swing_trajectory_1](./results/my_swing_example_disp_0.30_0.00_obstacle_0.06/my_swing_example_disp_0.30_0.00_obstacle_0.06.png)![swing_trajectory_1](./results/my_swing_example_disp_0.30_0.00_obstacle_0.08/my_swing_example_disp_0.30_0.00_obstacle_0.08.png)
