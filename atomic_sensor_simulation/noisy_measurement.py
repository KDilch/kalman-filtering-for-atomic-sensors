# -*- coding: utf-8 -*-
from noise import Noise, GaussianWhiteNoise
from atomic_sensor import Signal
import numpy as np


class NoisyDataGenerator(object):

    def __init__(self, signal, time_step, logs, noise=GaussianWhiteNoise(scalar_strength=1), length_data = 10000):
        if (noise is Noise) and (signal is Signal):
            self.__index_current = 0
            self.__noise = noise
            self.__signal = signal
            self.__time_step = time_step
            self.__all_data = np.zeros(length_data)
            self.__logs = logs
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
            self.__noise.step(time_step=self.__time_step)
            self.__all_data[0] = self.__noise.val + self.__signal.val
            self.__index_current += 1
        else:
            self.__logs.logging.warning('Step function is slow now. You exceeded declared data length.')
            self.__all_data.append(self.__noise.val + self.__signal.val)

    def generate(self):
        while self.__index_current < len(self.__all_data):
            self.step()
