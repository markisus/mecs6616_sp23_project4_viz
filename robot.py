import numpy as np
from math import ceil
import time
import threading


class Robot:

    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.state = np.zeros((dynamics.get_state_dim(), 1))
        self.action = np.zeros((dynamics.get_action_dim(), 1))
        self.t = 0
        self.goal = None

    def set_action(self, action):
        self.action = action

    def set_state(self, state):
        # first half is joint position, second half is joint velocity
        self.state = state

    def advance(self):
        state = self.state.copy()
        state = self.dynamics.advance(state, self.action)
        self.state = state
        self.t += self.dynamics.dt

    def get_t(self):
        return self.t

    def get_state(self):
        return self.state.copy()

    def set_t(self, t):
        self.t = t

    def reset(self):
        state = np.zeros((self.dynamics.get_state_dim(), 1))  # position + velocity
        state[0] = -np.pi / 2.0
        self.set_state(state)
        self.set_action(np.zeros((self.dynamics.get_action_dim(), 1)))
        self.set_t(0)
