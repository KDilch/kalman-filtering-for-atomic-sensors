#!/usr/bin/env python
# -*- coding: utf-8 -*-
from atomic_sensor_simulation.model.model import Model


class AtomicSensorModel(Model):

    def __init__(self,
                 F,
                 Q,
                 H,
                 R,
                 Gamma,
                 u,
                 z0,
                 dt
                 ):

        Model.__init__(self,
                       F=F,
                       H=H,
                       Q=Q,
                       R=R,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
