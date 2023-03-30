import math
import numpy as np
import time
import ast
from geometry import rot, xaxis, yaxis


class ArmDynamicsBase:
    """ Base class for arm dynamics """

    def __init__(self, num_links, link_mass, link_length, joint_viscous_friction=0.001, gravity=True, dt=0.001):
        self.num_links = num_links
        self.link_length = link_length
        self.link_mass = link_mass
        self.joint_viscous_friction = joint_viscous_friction
        self.gravity = gravity
        self.dt = dt
        self.link_lengths = np.full(self.num_links, float(self.link_length))
        self.link_masses = np.full(self.num_links, float(self.link_mass))
        self.link_inertias = np.zeros(self.num_links)
        self.model_loaded = False
        for i in range(0, self.num_links):
            self.link_inertias[i] = self.link_masses[i] * self.link_lengths[i] * self.link_lengths[i] / 12.0
        self.cpu_time_consumed = 0
        self.residue_limit = 1e-4
        self.residue_limit_flag = False

    def get_num_links(self):
        return self.num_links

    def get_link_lengths(self):
        return self.link_lengths

    def get_state_dim(self):
        return 2 * self.num_links

    def get_action_dim(self):
        return self.num_links

    def get_pos_0(self, state):
        return np.zeros((2, 1))

    def get_vel_0(self, state):
        return np.zeros((2, 1))

    def get_q(self, state):
        return state[0:self.num_links]

    def get_qd(self, state):
        return state[self.num_links: 2 * self.num_links]

    def compute_theta(self, q):
        return np.cumsum(q, axis=0)

    def compute_omega(self, qd):
        return np.cumsum(qd, axis=0)

    def compute_pos(self, pos_0, theta):
        pos_0 = pos_0.copy()
        theta = theta.copy()
        pos = np.zeros((2 * self.num_links, 1))
        pos[0:2] = pos_0
        for i in range(1, self.num_links):
            pos[2 * i:2 * (i + 1)] = pos[2 * (i - 1):2 * i] + np.dot(rot(theta[i - 1]),
                                                                     self.link_lengths[i - 1] * xaxis())
        return pos

    def compute_fk(self, state):
        pos_0 = self.get_pos_0(state)
        q = self.get_q(state)
        theta = self.compute_theta(q)
        pos = self.compute_pos(pos_0, theta)
        pos_ee = np.zeros((2,1))
        pos_ee[0] = pos[2*(self.num_links-1)]
        pos_ee[1] = pos[2*(self.num_links-1)+1]
        pos_ee = pos_ee + np.dot(rot(theta[self.num_links - 1]), self.link_lengths[self.num_links - 1] * xaxis())
        return pos_ee

    def compute_pos_com(self, pos, theta):
        pos_com = np.zeros((2 * self.num_links, 1))
        for i in range(0, self.num_links):
            pos_com[2 * i:2 * (i + 1)] = pos[2 * i:2 * (i + 1)] + np.dot(rot(theta[i]),
                                                                         0.5 * self.link_lengths[0] * xaxis())
        return pos_com

    def compute_vel(self, vel_0, omega, theta):
        vel_0 = vel_0.copy()
        theta = theta.copy()
        omega = omega.copy()
        vel = np.zeros((2 * self.num_links, 1))
        vel[0:2] = vel_0
        vel_world = np.zeros((2 * self.num_links, 1))
        vel_world[0:2] = (np.dot(rot(theta[0]), vel_0))
        for i in range(1, self.num_links):
            vel_world[2 * i:2 * (i + 1)] = vel_world[2 * (i - 1):2 * i] + (
                np.dot(rot(theta[i - 1]), omega[i - 1] * self.link_lengths[i - 1] * yaxis()))
            vel[2 * i:2 * (i + 1)] = np.dot(rot(-1.0 * theta[i]), vel_world[2 * i:2 * (i + 1)])
        return vel

    def compute_vel_ee(self, state):
        vel_0 = np.zeros((2,1))
        q = self.get_q(state)
        theta = self.compute_theta(q)
        qd = self.get_qd(state)
        omega = self.compute_omega(qd)
        vel = self.compute_vel(vel_0, omega, theta)
        return vel[-2:]

    def compute_vel_com(self, vel, omega):
        vel = vel.copy()
        omega = omega.copy()
        vel_com = np.zeros((2 * self.num_links, 1))
        for i in range(0, self.num_links):
            vel_com[2 * i + 0] = vel[2 * i + 0]
            vel_com[2 * i + 1] = vel[2 * i + 1] + omega[i] * 0.5 * self.link_lengths[i]

        return vel_com

    def advance(self, state, action):
        """ Forward simulation to compute new state given state and action """
        start_time = time.time()
        dt = self.dt
        new_state = self.dynamics_step(state, action, dt)
        self.cpu_time_consumed += time.time() - start_time
        return new_state

    def compute_energy(self, state):
        """ Computes lagrangian, kinetic engery, potential energy of the arm """
        pos_0 = self.get_pos_0(state)
        vel_0 = self.get_vel_0(state)
        q = self.get_q(state)
        qd = self.get_qd(state)
        theta = self.compute_theta(q)
        omega = self.compute_omega(qd)
        pos = self.compute_pos(pos_0, theta)
        pos_com = self.compute_pos_com(pos, theta)
        vel = self.compute_vel(vel_0, omega, theta)
        vel_com = self.compute_vel_com(vel, omega)
        # Sum energy for all links
        T = 0.0  # Kinetic energy
        V = 0.0  # Potential energy
        for i in range(0, self.num_links):
            T += 0.5 * self.link_masses[i] * (np.linalg.norm(vel_com[2 * i:2 * (i + 1)]) ** 2) \
                 + 0.5 * self.link_inertias[i] * (omega[i] ** 2)
            V += 9.8 * pos_com[2 * i + 1, 0] * self.link_masses[i]
        L = T - V  # Lagrangian
        return L, T, V

    def dynamics_step(self, state, action, dt):
        """ Advance simulation by one time step """
        raise NotImplementedError
