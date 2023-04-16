# Local Variables:
# python-indent: 2
# End:

import argparse
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_student import *
from collections import defaultdict
from imgui_plot_joints import *
from imgui_sdl_wrapper import ImguiSdlWrapper
from mpc import *
from overlayable import *
from robot import Robot
import imgui
import math
import numpy as np
import time

parser = argparse.ArgumentParser(description="Visualizer for Robotic Learning HW 4")
parser.add_argument("--use-student", action="store_true", default=False, help="Use the ArmDynamicsStudent instead of true dynamics for the MPC")

args = parser.parse_args()

# copied from score.py
# to not pull in unnecessary dependencies
def sample_goal():
  goal = np.zeros((2,1))
  r = np.random.uniform(low=0.05, high=1.95)
  theta = np.random.uniform(low=np.pi, high=2.0*np.pi)
  goal[0,0] = r * np.cos(theta)
  goal[1,0] = r * np.sin(theta)
  return goal

np.set_printoptions(suppress=True)
app = ImguiSdlWrapper("Simulator", 1280, 720)

use_student = args.use_student
use_student_cases = False

# Test cases for Part 1
test_cases = [
  (1, [ 0.38941834, -0.92106099]),
  (1, [-0.68163876, -0.73168887]),
  (2, [ 0.6814821,  -1.61185674]),
  (2, [-1.19286783, -1.28045552]),
  (3, [ 1.29444895, -2.36947292]),
  (3, [-2.10367746, -1.35075576]),
]
time_limit = 6.0

# Test cases for Part 2
# On the real Colab they are randomized
if use_student or use_student_cases:
  time_limit = 2.5
  test_cases = [
    (2, [-0.2758936731083355, -1.0043054984757163]),
    (2, [ 1.09857465, -0.26156757]),
    (2, [0.16797797397068537, -1.1833879134541971]),
    (2, [-0.27589367, -1.0043055 ]),
    (2, [-0.30039789, -1.00666561]),
    (2, [1.52142338, -1.07287136]),
    (2,[-0.56047203, -0.02274704] ),
    (2,[-0.21507777, -1.49818997] ),
  ]

dt = 0.01
run = False
goal = np.zeros((2,1))
t = 0
case_id = 0
need_restart = True
dynamics_student = None
if use_student:
  import torch
  dynamics_student = ArmDynamicsStudent(
      num_links=2,
      link_mass=0.1,
      link_length=1,
      joint_viscous_friction=0.1,
      dt=dt)
  # device = torch.device('cuda')
  device = torch.device('cpu')
  dynamics_student.init_model("dynamics.pth", num_links=2, time_step=dt, device=device)

while app.running:
    if need_restart:
      t = 0.0
      step = 0

      if case_id != 0:
        num_links, case_goal = test_cases[case_id-1]
        goal_x, goal_y = case_goal
      else:
        # randomized mode
        num_links = 2
        goal_x, goal_y = sample_goal().flatten()

      goal[0,0] = goal_x
      goal[1,0] = goal_y
      dynamics_teacher = ArmDynamicsTeacher(
          num_links=num_links,
          link_mass=0.1,
          link_length=1,
          joint_viscous_friction=0.1,
          dt=dt)

      arm = Robot(dynamics_teacher)
      arm.reset()
      controller = MPC()
      action = np.zeros((num_links, 1))
      need_restart = False
  
    app.main_loop_begin()

    state = arm.get_state()    
    pos_ee = dynamics_teacher.compute_fk(state)
    dist = np.linalg.norm(goal-pos_ee)
    vel_ee = np.linalg.norm(arm.dynamics.compute_vel_ee(state))

    imgui.begin("Debug", True)
    need_restart = imgui.button("restart") or need_restart

    imgui.text(f"Case id 0 => Randomized goal")
    case_changed, case_id = imgui.slider_int("Case Id", case_id, 0, len(test_cases))
    need_restart = need_restart or case_changed
    
    _, run = imgui.checkbox("Run", run)


    imgui.text(f'At time {t:0.3}: Distance to goal: {dist:0.3}, Velocity of end effector: {vel_ee:0.3}')
    _, goal_x = imgui.slider_float("goal x", goal_x, -3, 3)
    _, goal_y = imgui.slider_float("goal y", goal_y, -3, 3)

    goal[0,0] = goal_x
    goal[1,0] = goal_y

    display_width = imgui.get_window_width() - 10
    origin_plot = to_ndc(np.zeros((2,1))).flatten()
    goal_plot = to_ndc(goal).flatten()
    overlayable = draw_overlayable_rectangle(1, 1, display_width)
    overlay_circle(overlayable, goal_plot[0], goal_plot[1], 0.015, imgui.get_color_u32_rgba(0,1,0,1), 1)

    qual1 = int(dist < 0.1)
    qual2 = int(vel_ee < 0.5)
    num_quals = qual1 + qual2
    if num_quals < 1:
      arm_color = imgui.get_color_u32_rgba(1,0,0,1)
    elif num_quals < 2:
      arm_color = imgui.get_color_u32_rgba(0,0,1,1)
    else:
      arm_color = imgui.get_color_u32_rgba(0,1,0,1)

    plot_joints(overlayable, arm.dynamics, state, arm_color)
    imgui.end()

    if run and step % controller.control_horizon == 0:
      if use_student:
        action = controller.compute_action(dynamics_student, state, goal, action)
      else:
        action = controller.compute_action(arm.dynamics, state, goal, action)
      arm.set_action(action)

    if run:
      arm.advance()
      t += dt

    if t-dt <= time_limit and t > time_limit and run:
      run = False

    app.main_loop_end()
app.destroy()    


