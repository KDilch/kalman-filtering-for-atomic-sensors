#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import KalmanFilter
import numpy as np


class AtomicSensorModel(object):
    def __init__(self, F, Phi, initial_state, dim_z, dt, scalar_strenght_y, scalar_strength_j, scalar_strength_q, g_d_COUPLING_CONST=1.):
        self.x = initial_state
        self.dim_z = dim_z
        self.F = F
        self.Phi = Phi
        self.Q = np.array([[scalar_strength_j, 0.], [0., scalar_strength_q]])
        self.Q_delta = np.dot(np.dot(self.Phi, self.Q), self.Phi.transpose())*dt
        self.P = np.dot(np.dot(self.Phi, self.Q), self.Phi.transpose()) + self.Q_delta
        self.H = np.array([[g_d_COUPLING_CONST, 0.]])
        self.R = [[scalar_strenght_y**2/dt]]
        self.filterpy = self.initialize_filterpy()

    def initialize_filterpy(self):
        filterpy = KalmanFilter(dim_x=len(self.x), dim_z=self.dim_z)
        filterpy.x = self.x
        filterpy.F = self.F
        filterpy.P = self.P
        filterpy.Q = self.Q_delta
        filterpy.H = self.H
        filterpy.R = self.R
        return filterpy

    def predict_step(self):
        x_exact = np.dot(self.F, self.x)


    def update_step(self):
        pass

    def batch_filter(self):
        pass

    def step(self):
        pass
