#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import ExtendedKalmanFilter
from scipy.integrate import odeint, simps
import sympy
import numpy as np
from atomic_sensor_simulation.filter_model.model import Model
from atomic_sensor_simulation.utilities import eval_matrix_of_functions


class Extended_KF(Model):

    def __init__(self,
                 Q,
                 H,
                 R,
                 Gamma,
                 u,
                 z0,
                 dt,
                 num_terms,
                 time_arr
                 ):

        Model.__init__(self,
                       Q=Q,
                       R=R,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
        self.H = H
        self.H_inverse = np.linalg.pinv(np.array(H, dtype=np.float))
        self.x0, self.P0 = self.calculate_x0_and_P0(z0)
        self.x0=np.array([2,2,0.1])
        self.num_terms = num_terms
        self.dim_x = len(self.x0)
        self.time_arr = time_arr

    def hx(self, x):
        return self.H.dot(x)

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
        print(x0)
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
        self.fxu = lambda t: np.array([np.cos(self.x[2] * t) * self.x[0] - np.sin(self.x[2] * t) * self.x[1],
                                       np.sin(self.x[2] * t) * self.x[0] + np.cos(self.x[2] * t) * self.x[1],
                                       self.x[2]])
        self.F = lambda t: np.array([[np.cos(self.x[2] * t), -np.sin(self.x[2] * t),
                                      -np.sin(self.x[2] * t) * self.x[0] * t - np.cos(self.x[2] * t) * self.x[
                                          1] * t],
                                     [np.sin(self.x[2] * t), np.cos(self.x[2] * t),
                                      np.cos(self.x[2] * t) * self.x[0] * t - np.sin(self.x[2] * t) * self.x[
                                          1] * t],
                                     [0, 0, 1]])

    def predict(self, u=0):
        self.x = self.move()
        self.t += self.dt
        self.P = np.dot(self.F(self.t), self.P).dot(self.F(self.t).T) + self.Q
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def move(self):
        return self.fxu(self.t)