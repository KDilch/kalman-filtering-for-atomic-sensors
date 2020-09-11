#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from atomic_sensor_simulation.filter_model.model import Model
from atomic_sensor_simulation.utilities import eval_matrix_of_functions


class Unscented_KF(Model):

    def __init__(self,
                 F,
                 Q,
                 H,
                 R,
                 R_delta,
                 Gamma,
                 u,
                 z0,
                 dt,
                 x0,
                 P0
                 ):

        Model.__init__(self,
                       Q=Q,
                       R=R,
                       R_delta=R_delta,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
        self.F = F
        self.fx = self.compute_fx_at_time_t(0)
        self.H = H
        self.points = MerweScaledSigmaPoints(4, alpha=.00001, beta=2., kappa=0.) #TODO figure out the factors
        self.x0 = x0
        self.P0 = P0
        self.dim_x = len(self.x0)

    def compute_fx_at_time_t(self, t):
        F_t = eval_matrix_of_functions(self.F, t)

        def fx(x):
            return x + F_t.dot(x) 

        return fx

    def set_Q(self, Q):
        self.Q = Q

    def hx(self, x):
        return self.H.dot(x)

    def initialize_filterpy(self):
        self._logger.info('Initializing Linear Kalman Filter (filtepy)...')
        filterpy = UnscentedKalmanFilter(dim_x=self.dim_x,
                                         dim_z=self.dim_z,
                                         dt=self.dt,
                                         fx=self.fx,
                                         hx=self.hx,
                                         points=self.points)
        filterpy.x = self.x0.T
        filterpy.P = self.P0
        filterpy.Q = self.Q
        filterpy.R = self.R_delta

        return filterpy
