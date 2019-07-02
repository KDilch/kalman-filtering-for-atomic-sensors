#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from atomic_sensor_simulation.model.model import Model


class PosVelModel(Model):

    def __init__(self,
                 F,
                 Gamma,
                 u,
                 z0,
                 dt,
                 ):
        H = np.array([[1. / 0.3048, 0., 0., 0.], [0., 0., 1. / 0.3048, 0.]], dtype='float64')
        R = np.eye(2) * 5.
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
        Q = block_diag(q, q)

        Model.__init__(self, F=F, H=H, Q=Q, R=R, Gamma=Gamma, u=u, z0=z0, dt=dt)