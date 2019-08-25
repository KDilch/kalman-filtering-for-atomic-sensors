#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from atomic_sensor_simulation.filter_model.model import Model


class Unscented_KF(Model):

    def __init__(self,
                 fx,
                 Q,
                 hx,
                 R,
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
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
        self.fx = fx
        self.hx = hx
        self.points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1) #TODO figure out the factors
        #TODO figure out what Q_delta is
        self.x0 = x0
        self.P0 = P0
        self.dim_x = len(self.x0)

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
