#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.linalg import expm

from atomic_sensor_simulation.homemade_kalman_filter.homemade_kf import HomeMadeKalmanFilter
from atomic_sensor_simulation.utilities import integrate_matrix_of_functions, eval_matrix_of_functions

class AtomicSensorModel(object):

    def __init__(self,
                 F,
                 Gamma,
                 u,
                 z0,
                 dt,
                 scalar_strength_z,
                 scalar_strength_jy,
                 scalar_strength_jz,
                 scalar_strength_qp,
                 scalar_strength_qq,
                 g_d_COUPLING_CONST=1.):
        self.dt = dt
        self.F = F
        self.Phi_delta = self.compute_Phi_delta(from_time=0)
        self.Q = np.array([[scalar_strength_jy**2, 0., 0., 0.],
                           [0., scalar_strength_jz**2, 0., 0.],
                           [0., 0., scalar_strength_qp, 0.],
                           [0., 0., 0., scalar_strength_qq]])
        self.Q_delta = np.dot(np.dot(self.Phi_delta, self.Q), self.Phi_delta.transpose()) * dt
        self.H = np.array([[0., g_d_COUPLING_CONST, 0.,  0.]])
        self.H_inverse = np.linalg.pinv(self.H)
        self.R_delta = [[scalar_strength_z ** 2 / dt]]

        self.x0, self.P0 = self.calculate_x0_and_P0(z0)
        print(self.x0, "initialize x0")
        self.dim_x = len(self.x0)
        self.dim_z = len(z0)

        self.Gamma_control_transition_matrix = Gamma
        self.u_control_vec = u

    def calculate_x0_and_P0(self, z0):
        x0 = np.dot(self.H_inverse, z0)
        cov_x0 = self.Q + np.dot(np.dot(self.H_inverse, self.R_delta), np.transpose(self.H_inverse))
        return x0, cov_x0

    def compute_Phi_delta(self, from_time):
        return expm(integrate_matrix_of_functions(self.F, from_time, from_time+self.dt))

    def initialize_filterpy(self):
        filterpy = KalmanFilter(dim_x=len(self.x0), dim_z=self.dim_z)
        filterpy.x = self.x0
        filterpy.P = self.P0

        filterpy.F = self.Phi_delta
        filterpy.Q = self.Q_delta
        filterpy.H = self.H
        filterpy.R = self.R_delta
        return filterpy

    def initialize_homemade_filter(self):
        return HomeMadeKalmanFilter(x0=self.x0,
                                    P0=self.P0,
                                    Phi_delta=self.Phi_delta,
                                    Q_delta=self.Q_delta,
                                    H=self.H,
                                    R_delta=self.R_delta,
                                    Gamma=self.Gamma_control_transition_matrix,
                                    u=self.u_control_vec)
