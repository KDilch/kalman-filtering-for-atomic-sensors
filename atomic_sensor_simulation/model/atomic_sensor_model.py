#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import KalmanFilter
from atomic_sensor_simulation.homemade_kalman_filter.homemade_kf import HomeMadeKalmanFilter
import numpy as np


class AtomicSensorModel(object):

    def __init__(self, F, Phi, z0, dt, scalar_strength_z, scalar_strength_j, scalar_strength_q, g_d_COUPLING_CONST=1.):
        self.F = F
        self.Phi_delta = Phi
        self.Q = np.array([[scalar_strength_j**2, 0.], [0., scalar_strength_q**2]])
        self.Q_delta = np.dot(np.dot(self.Phi_delta, self.Q), self.Phi_delta.transpose()) * dt
        self.H = np.array([[g_d_COUPLING_CONST, 0.]])
        self.H_inverse = np.linalg.pinv(self.H)
        self.R_delta = [[scalar_strength_z ** 2 / dt]]

        self.x0, self.P0 = self.calculate_x0_and_P0(z0)
        self.dim_x = len(self.x0)
        self.dim_z = len(z0)
        self.dt = dt

    def calculate_x0_and_P0(self, z0):
        x0 = np.dot(self.H_inverse, z0)
        cov_x0 = self.Q + np.dot(np.dot(self.H_inverse, self.R_delta), np.transpose(self.H_inverse))
        return x0, cov_x0

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
                                    R_delta=self.R_delta)
