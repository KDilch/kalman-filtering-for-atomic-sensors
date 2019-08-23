#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from atomic_sensor_simulation.linear_KF_model.model import Model


class PosVelModel(Model):

    def __init__(self,
                 F,
                 Gamma,
                 u,
                 z0,
                 dt,
                 ):


        Model.__init__(self, F=F, H=H, Q=Q, R=R, Gamma=Gamma, u=u, z0=z0, dt=dt)