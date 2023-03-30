from arm_dynamics_base import ArmDynamicsBase
import numpy as np
from geometry import rot, xaxis, yaxis


class ArmDynamicsTeacher(ArmDynamicsBase):

    def idx_f(self, i):
        """ Returns index of f in vars"""
        return 2 * i

    def idx_a(self, i):
        """ Returns index of a in vars"""
        return 2 * self.num_links + 2 * i

    def idx_omdot(self, i):
        """ Returns index of omdot in vars"""
        return 2 * self.num_links + 2 * self.num_links + i

    def idx_f_eqbm(self, i):
        """ Returns index of force equilibrium constraints in constraint matrix"""
        return self.idx_f(i)

    def idx_tau_eqbm(self, i):
        """ Returns index of torque equilibrium constraints in constraint matrix"""
        return 2 * self.num_links + i

    def num_var(self):
        return 2 * self.num_links + 2 * self.num_links + self.num_links

    def constraint_matrices(self, state, action):
        """ Contructs the constraint matrices from state """
        # Computes variables dependent on state required to construct constraint matrices
        num_vars = self.num_var()
        q = self.get_q(state)
        theta = self.compute_theta(q)
        qd = self.get_qd(state)
        omega = self.compute_omega(qd)
        vel_0 = self.get_vel_0(state)
        vel = self.compute_vel(vel_0, omega, theta)
        vel_com = self.compute_vel_com(vel, omega)

        left_hand = None
        right_hand = None

        # Force equilibrium constraints
        for i in range(0, self.num_links):
            cl = np.zeros((2, num_vars))
            cl[0:2, self.idx_f(i):self.idx_f(i + 1)] = -1 * np.eye(2)
            cl[0:2, self.idx_a(i):self.idx_a(i + 1)] = -1 * self.link_masses[i] * np.eye(2)
            cl[1, self.idx_omdot(i)] = -1 * 0.5 * self.link_lengths[i] * self.link_masses[i]
            if i < self.num_links - 1:
                cl[0:2, self.idx_f(i + 1):self.idx_f(i + 2)] = rot(q[i + 1])
            cr = np.zeros((2, 1))
            # gravity
            if self.gravity:
                cr = cr + (-1 * 9.8 * self.link_masses[i]) * (np.dot(rot(-1 * theta[i]), (-1 * yaxis())))
            # centrifugal force
            cr[0] = cr[0] + (-1) * (omega[i] * omega[i] * 0.5 * self.link_lengths[i] * self.link_masses[i])
            if i == 0:
                left_hand = cl
                right_hand = cr
            else:
                left_hand = np.concatenate((left_hand, cl))
                right_hand = np.concatenate((right_hand, cr))

        # Torque equilibrium constraints
        for i in range(0, self.num_links):
            cl = np.zeros((1, num_vars))
            # the y component of the force
            cl[0, self.idx_f(i) + 1] = self.link_lengths[i] * 0.5
            # inertial torque
            cl[0, self.idx_omdot(i)] = -1 * self.link_inertias[i]
            if i < self.num_links - 1:
                # the y component
                cl[0, self.idx_f(i + 1):self.idx_f(i + 2)] = self.link_lengths[i] * 0.5 * rot(q[i + 1])[1, :]
            left_hand = np.concatenate((left_hand, cl))
            cr = np.zeros((1, 1))
            right_hand = np.concatenate((right_hand, cr))
            # viscous friction depends on the mode, implemented in ArmDynamics & SnakeDynamics

        # Linear acceleration constraints
        for i in range(1, self.num_links):
            cl = np.zeros((2, num_vars))
            cl[0:2, self.idx_a(i):self.idx_a(i + 1)] = -1 * np.eye(2)
            cl[0:2, self.idx_a(i - 1):self.idx_a(i)] = rot(-1 * q[i])
            cl[0:2, self.idx_omdot(i - 1):self.idx_omdot(i)] = self.link_lengths[i - 1] * (
                np.dot(rot(-1 * q[i]), (1 * yaxis())))
            left_hand = np.concatenate((left_hand, cl))
            cr = -1 * self.link_lengths[i - 1] * omega[i - 1] * omega[i - 1] * (np.dot(rot(-1 * q[i]), (-1 * xaxis())))
            right_hand = np.concatenate((right_hand, cr))

        assert left_hand.shape == (self.num_var() - 2, self.num_var())
        assert right_hand.shape == (self.num_var() - 2, 1)

        # Joint viscous friction
        for i in range(self.num_links):
            right_hand[self.idx_tau_eqbm(i)] += qd[i] * self.joint_viscous_friction

        # Linear acceleration of joint-0 must be zero
        cl = np.zeros((2, self.num_var()))
        cl[0:2, self.idx_a(0):self.idx_a(1)] = np.eye(2)
        left_hand = np.concatenate((left_hand, cl))
        cr = np.zeros((2, 1))
        right_hand = np.concatenate((right_hand, cr))

        assert left_hand.shape == (5 * self.num_links, 5 * self.num_links)
        assert right_hand.shape == (5 * self.num_links, 1)

        # Apply torques 
        tau = action
        for i in range(self.num_links):
            right_hand[self.idx_tau_eqbm(i), 0] += (tau[i + 1] if i < self.num_links - 1 else 0.0) - tau[i]

        return left_hand, right_hand

    def solve(self, left_hand, right_hand):
        """ Solves the constraint matrices to compute accelerations """
        x = np.linalg.solve(left_hand, right_hand)
        self.residue = np.linalg.norm(np.dot(left_hand, x) - right_hand) / self.num_var()
        residue = np.linalg.norm(np.dot(left_hand, x) - right_hand) / self.num_var()
        if residue > self.residue_limit:
            print('cannot solve, residue {} exceeds limit {}'.format(residue, self.residue_limit))
            self.residue_limit_flag = True
        a = x[self.idx_a(0):self.idx_a(self.num_links)]
        omdot = x[self.idx_omdot(0):self.idx_omdot(self.num_links)]
        qdd = omdot.copy()
        for i in range(self.num_links - 1, 0, -1):
            qdd[i] -= qdd[i - 1]
        return a, qdd

    def dynamics_step(self, state, action, dt):
        """ Forward simulation using Euler method """
        left_hand, right_hand = self.constraint_matrices(state, action)
        a, qdd = self.solve(left_hand, right_hand)
        new_state = self.integrate_euler(state, a, qdd, dt)
        return new_state

    def integrate_euler(self, state, a, qdd, dt):
        """ Integrates using Euler method """
        # Compute state dependent variables needed for integration
        pos_0 = self.get_pos_0(state)
        vel_0 = self.get_vel_0(state)
        q = self.get_q(state)
        qd = self.get_qd(state)
        theta = self.compute_theta(q)

        qd_new = qd + qdd * dt
        q_new = q + 0.5 * (qd + qd_new) * dt

        theta_new = self.compute_theta(q_new)
        vel_0_new = np.dot(rot(theta[0] - theta_new[0]), (vel_0 + a[0:2] * dt))
        pos_0_new = pos_0 + 0.5 * (np.dot(rot(theta[0]), vel_0) + np.dot(rot(theta_new[0]), vel_0_new)) * dt

        new_state = np.vstack([q_new, qd_new])

        return new_state
