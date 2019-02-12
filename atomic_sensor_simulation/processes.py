# -*- coding: utf-8 -*-
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class OrnsteinUhlenbeckProcess(object):

    def __init__(self, initial_state, correlation_time, scalar_strength):
        self.__correlation_time = correlation_time
        self.__state = initial_state
        self.__noise = GaussianWhiteNoise(scalar_strength)
        self.__val = None

    @property
    def val(self):
        if self.__val is not None:
            return self.__val
        else:
            Warning('Not current value defined for quadrature. You need to step first.')

    def step(self, time_step):
        self.__noise.__step(time_step)
        self.__val = -self.__correlation_time*self.__state * time_step + self.__noise.val
        return
