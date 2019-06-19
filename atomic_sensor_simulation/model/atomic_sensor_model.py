#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np


class AtomicSensorModel(object):
    def __init__(self, F, Phi, z0, dim_z, dt, scalar_strenght_y, scalar_strength_j, scalar_strength_q, g_d_COUPLING_CONST=1.):
        self.dim_z = dim_z
        self.scalar_strenght_y = scalar_strenght_y
        self.scalar_strenght_q = scalar_strength_q
        self.scalar_strenght_j = scalar_strength_j
        self.dt = dt
        self.F = F
        self.Phi = Phi
        self.Q = np.array([[scalar_strength_j**2, 0.], [0., scalar_strength_q**2]])
        self.Q_delta = np.dot(np.dot(self.Phi, self.Q), self.Phi.transpose())*dt
        # self.Q_lib = Q_discrete_white_noise(2, dt=dt, var=scalar_strength_q)
        self.H = np.array([[g_d_COUPLING_CONST, 0.]])
        self.R = [[scalar_strenght_y**2/dt]]
        self.x, self.P = self.calculate_x0_and_covx0(z0)
        self.filterpy = self.initialize_filterpy()


    def initialize_filterpy(self):
        filterpy = KalmanFilter(dim_x=len(self.x), dim_z=self.dim_z)
        filterpy.x = self.x
        filterpy.P = self.P

        filterpy.F = self.Phi
        filterpy.Q = self.Q_delta
        filterpy.H = self.H
        filterpy.R = self.R
        return filterpy

    def calculate_x0_and_covx0(self, z0):
        H_inverse = np.linalg.pinv(self.H)
        Q = np.array([[self.scalar_strenght_j ** 2, 0.], [0., self.scalar_strenght_q ** 2]])
        R_delta = [[self.scalar_strenght_y ** 2 / self.dt]]
        x0 = np.dot(H_inverse, z0)
        cov_x0 = Q + np.dot(np.dot(H_inverse, R_delta), np.transpose(H_inverse))
        return x0, cov_x0


class HomeMadeKalmanFilter(object):

    def __init__(self, dimx, dimz, x0, error_x0, Phi_delta, R_delta, G=None, Q_delta=None, H=None):
        self.x = x0
        self.P = error_x0
        self.Phi_delta = Phi_delta
        self.dimx = dimx
        self.dimz = dimz
        self.Q_delta = Q_delta
        self.H = H
        self.R_delta = R_delta
        print(x0, error_x0)
        if G:
            self.G = G
        else:
            self.G = np.identity(self.dimx)

    def predict(self):
        x = np.dot(self.Phi_delta, self.x)
        P = np.dot(np.dot(self.Phi_delta, self.P), np.transpose(self.Phi_delta)) + self.Q_delta
        self.x = x
        self.P = P
        print("Predict, x, cov", self.x, self.P)

    def update(self, z):
        z_est = np.dot(self.H, self.x)
        y = z - z_est
        S = self.R_delta + np.dot(np.dot(self.H, self.P), np.transpose(self.H))
        S_inverse = np.linalg.inv(S)
        K = np.dot(np.dot(self.P, np.transpose(self.H)), S_inverse)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.identity(2) - np.dot(K, self.H)), self.P)
        print("Update, x, cov", self.x, self.P)
