# Local Variables:
# python-indent: 2
# End:

from arm_dynamics_teacher import ArmDynamicsTeacher
from collections import defaultdict
from imgui_sdl_wrapper import ImguiSdlWrapper
from overlayable import *
from imgui_plot_joints import *
from robot import Robot
import imgui
import math
import numpy as np
import time
import os
import pickle
 
np.set_printoptions(suppress=True)

with open(os.path.join('dataset/data.pkl'), "rb" ) as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y']

state_dim = Y.shape[1]
action_dim = Y.shape[1] - X.shape[1]
num_links = state_dim // 2

dynamics = ArmDynamicsTeacher(
    num_links=num_links,
    link_mass=0.1, # guess
    link_length=1, # guess
    joint_viscous_friction=0.1, # guess
    dt=0.01)

app = ImguiSdlWrapper("Data Viewer", 1280, 720)

data_idx = 0
while app.running:
    app.main_loop_begin()
    imgui.begin("Viz", True)

    _, data_idx = imgui.slider_int("data#", data_idx, 0, X.shape[0]-1)

    x = X[data_idx,:]
    y = Y[data_idx,:]

    imgui.text(f"x: {x.T}")
    imgui.text(f"y: {y.T}")

    display_width = imgui.get_window_width() - 10

    origin_plot = to_ndc(np.zeros((2,1))).flatten()
    overlayable = draw_overlayable_rectangle(1, 1, display_width)
    overlay_circle(overlayable, origin_plot[0], origin_plot[1], 0.01, imgui.get_color_u32_rgba(1,1,1,1), 1)

    arm_color = imgui.get_color_u32_rgba(1,1,1,1)
    plot_joints(overlayable, dynamics, x, arm_color)
    
    imgui.end()

    app.main_loop_end()
app.destroy()    

