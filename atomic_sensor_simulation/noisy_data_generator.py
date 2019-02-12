# -*- coding: utf-8 -*-
from atomic_sensor_simulation.noise import Noise, GaussianWhiteNoise
from atomic_sensor_simulation.state import Signal
import numpy as np
import logging


class NoisyDataGenerator(object):

    def __init__(self,
                 signal,
                 time_step,
                 noise=GaussianWhiteNoise(initial_value=1, scalar_strength=1, dt=1),
                 data_length=10000,
                 logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        if (noise is Noise) and (signal is Signal):
            self.__index_current = 0
            self.__noise = noise
            self.__signal = signal
            self.__time_step = time_step
            self.__all_data = np.zeros(data_length)
        else:
            raise TypeError('Class NoisyDataGenerator initialized with parameters of a wrong type.')

    @property
    def all_data(self):
        return self.__all_data

    @property
    def val_current(self):
        return self.__all_data[self.__index_current]

    def step(self):
        if self.__index_current < len(self.__all_data):
            self.__noise.__step(time_step=self.__time_step)
            self.__all_data[0] = self.__noise.val + self.__signal.val
            self.__index_current += 1
        else:
            self.__all_data.append(self.__noise.val + self.__signal.val)

    def generate(self):
        while self.__index_current < len(self.__all_data):
            self.step()


