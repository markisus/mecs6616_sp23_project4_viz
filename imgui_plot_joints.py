import numpy as np
from overlayable import *
from geometry import rot

def to_ndc(vec2):
  out = vec2/10
  out[1,:] *= -1
  out += np.array([[0.5, 0.3]]).T
  return out

def plot_joints(overlayable, dynamics, state, color):
  p = np.zeros((2, 1))
  R = np.eye(2)
  q = dynamics.get_q(state)
  pos_0 = dynamics.get_pos_0(state)
  num_links = dynamics.get_num_links()
  link_lengths = dynamics.get_link_lengths()
  off_x, off_y = pos_0[0], pos_0[1]

  for i in range(0, num_links):
      R = np.dot(R, rot(q[i]))
      l = np.zeros((2, 1))
      l[0, 0] = link_lengths[i]
      p_next = p + np.dot(R, l)

      plot_p = to_ndc(p).flatten()
      plot_p_next = to_ndc(p_next).flatten()

      overlay_line(overlayable, plot_p[0], plot_p[1], plot_p_next[0], plot_p_next[1],
                   color, 1)
      overlay_circle(overlayable, plot_p[0], plot_p[1], 0.01, color, 1)
      p = p_next

  overlay_circle(overlayable, plot_p_next[0], plot_p_next[1], 0.01, color, 1)
