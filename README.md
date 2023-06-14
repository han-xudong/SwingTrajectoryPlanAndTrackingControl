# SwingTrajectoryPlanAndTrackingControl

In this project, we focus on the planning and control for the swing trajectories.

The codes are based on: https://xbpeng.github.io/projects/Robotic_Imitation/index.html

## Getting Started

We use this repository with Python 3.7 or Python 3.8 on Ubuntu, MacOS and Windows.

- Install MPC extension (Optional) `python3 setup.py install --user`

Install dependencies:

- Install MPI: `sudo apt install libopenmpi-dev`
- Install requirements: `pip3 install -r requirements.txt`

and it should be good to go.

## Finding a feasible trajectory connecting the initial and final positions
We give a method to find a feasible swing trajectory when given an initial and final position of the foot tip.
