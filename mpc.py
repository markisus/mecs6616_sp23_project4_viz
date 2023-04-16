# Local Variables:
# python-indent: 2
# End:

import numpy as np

class MPC:
  def __init__(self):
    self.control_horizon = 10

  def compute_action(self, dynamics, state, goal, action):
    # dummy impl
    return np.random.uniform(-5.0, 5.0, (dynamics.get_num_links(), 1))
