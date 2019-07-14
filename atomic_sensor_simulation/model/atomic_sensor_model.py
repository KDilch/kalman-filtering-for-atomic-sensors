#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from atomic_sensor_simulation.model.model import Model


class AtomicSensorModel(Model):

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
        Q = np.array([[scalar_strength_jy**2, 0., 0., 0.],
                           [0., scalar_strength_jz**2, 0., 0.],
                           [0., 0., scalar_strength_qp**2, 0.],
                           [0., 0., 0., scalar_strength_qq**2]])
        H = np.array([[0., g_d_COUPLING_CONST, 0.,  0.]])
        R_delta = [[scalar_strength_z ** 2 / dt]]
        Model.__init__(self,
                       F=F,
                       H=H,
                       Q=Q,
                       R=R_delta,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
