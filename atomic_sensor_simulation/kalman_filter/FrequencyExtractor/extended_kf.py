#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import ExtendedKalmanFilter
from scipy.integrate import odeint, simps
import sympy
import numpy as np
from atomic_sensor_simulation.utilities import eval_matrix_of_functions
class Model():
    pass

class Extended_KF(Model):

    def __init__(self,
                 Q,
                 H,
                 R_delta,
                 Gamma,
                 u,
                 z0,
                 dt,
                 num_terms,
                 time_arr
                 ):

        Model.__init__(self,
                       Q=Q,
                       R_delta=R_delta,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
        self.H = H
        self.H_inverse = np.linalg.pinv(np.array(H, dtype=np.float))
        self.x0, self.P0 = self.calculate_x0_and_P0(z0)
        self.num_terms = num_terms
        self.dim_x = len(self.x0)
        self.time_arr = time_arr

    def compute_Q_delta_sympy(self, from_time, F_0, num_terms=30):
        def dPhidt(Phi, t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self._F, t), dtype=float),
                                     np.reshape(Phi, (4, 4))), 16)

        t = np.linspace(from_time, from_time + self.dt, num=num_terms)  # times to report solution
        Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, 16), t, full_output=True)
        # Numerical
        Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (4, 4)) for i in range(len(Phi_deltas))]
        Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
        integrands = np.array(
            [np.dot(np.dot(a, self.Q), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
        int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23, int30, int31, int32, int33 = map(
            list, zip(*integrands.reshape(*integrands.shape[:1], -1)))
        integrand_split = [int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23, int30,
                           int31, int32, int33]
        # calculate integral numerically using simpsons rule
        self.Q_delta = np.reshape(np.array([simps(i, t) for i in integrand_split]), (4, 4))
        return np.reshape(np.array([simps(i, t) for i in integrand_split]), (4, 4))

    def hx(self, x):
        a = np.dot(self.H, x)
        return np.dot(self.H, x)

    def HJacobianat(self, x):
        return self.H

    def initialize_filterpy(self, **kwargs):
        self._logger.info('Initializing Extended Kalman Filter (filtepy)...')
        filterpy = FrequencyExtractorEKF(dim_x=self.dim_x,
                                   dim_z=self.dim_z,
                                   num_terms=self.num_terms,
                                   dt=self.dt,
                                   x0=self.x0,
                                   P0=self.P0,
                                   state_vec=self.x0,
                                   **kwargs)
        filterpy.x = self.x0
        filterpy.P = self.P0
        filterpy.Q = self.Q
        filterpy.R = self.R_delta

        return filterpy

    def calculate_x0_and_P0(self, z0):
        x0 = np.dot(self.H_inverse, z0)
        x0[1] = 1.
        print("x0 - filter:", x0)
        cov_x0 = self.Q + np.dot(np.dot(self.H_inverse, self.R_delta), np.transpose(self.H_inverse))
        print(cov_x0)
        return x0, cov_x0

class FrequencyExtractorEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, x0, P0, state_vec, **kwargs):
        ExtendedKalmanFilter.__init__(self, dim_x, dim_z)
        self.dt = dt
        self.t = 0
        self.x0 = x0
        self.P0 = P0
        self.x = state_vec
        self.fxu = lambda t, x: np.array([np.cos(x[2]*self.dt) * x[0] - np.sin(x[2]*self.dt) * x[1],
                                          np.sin(x[2]*self.dt) * x[0] + np.cos(x[2]*self.dt) * x[1],
                                          x[2]])
        # self.F = lambda t, x: np.array([[np.cos(x[2]*self.dt), -np.sin(x[2]*self.dt), -np.sin(x[2]*self.dt)*x[0]*self.dt-np.cos(x[2]*self.dt)*x[1]*self.dt],
        #                                 [np.sin(x[2]*self.dt), np.cos(x[2]*self.dt), np.cos(x[2]*self.dt)*x[0]*self.dt-np.sin(x[2]*self.dt)*x[1]*self.dt],
        #                                 [0, 0, 1]])
    def F_func(self, x):
        return np.array([[np.cos(x[2]*self.dt), -np.sin(x[2]*self.dt), -np.sin(x[2]*self.dt)*x[0]*self.dt-np.cos(x[2]*self.dt)*x[1]*self.dt],
                                        [np.sin(x[2]*self.dt), np.cos(x[2]*self.dt), np.cos(x[2]*self.dt)*x[0]*self.dt-np.sin(x[2]*self.dt)*x[1]*self.dt],
                                        [0, 0, 1]], dtype=float)

    def predict(self, u=0):
        self.x = self.move()
        self.t += self.dt
        self.P = np.dot(np.dot(self.F_func(self.x), self.P), np.transpose(self.F_func(self.x))) + self.Q
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def move(self):
        return self.fxu(self.t, self.x)