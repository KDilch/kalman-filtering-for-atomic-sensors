#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import KalmanFilter
import numpy as np


class AtomicSensorModel(object):
    def __init__(self, F, initial_state, dim_z):
        self.x = initial_state
        self.dim_z = dim_z
        self.F = F
        self.P = np.eye(2) * 50.
        self.Q = np.array([[0, 0], [0, 0]])
        self.H = np.array([[0., 1.]])
        self.R = [[1.]]
        self.filterpy = self.initialize_filterpy()

    def initialize_filterpy(self):
        filterpy = KalmanFilter(dim_x=len(self.x), dim_z=self.dim_z)
        filterpy.x = self.x
        filterpy.F = self.F
        filterpy.P = self.P
        filterpy.Q = self.Q
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
