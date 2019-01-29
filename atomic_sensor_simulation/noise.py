# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import randn  # white noise
from abc import ABC, abstractmethod  # abstract class


class Noise(ABC):
    @abstractmethod
    def step(self, time_step):
        raise NotImplementedError

    @property
    @abstractmethod
    def val(self):
        raise NotImplementedError


class GaussianWhiteNoise(Noise):
    def __init__(self, scalar_strength, mean=0):
        self.__mean = mean
        self.__scalar_strength = scalar_strength  # determined experimentally
        self.__val = None

    @property
    def val(self):
        if self.__val is not None:
            return self.__val
        else:
            Warning('Not current value defined for gaussian noise. You need to step first.')

    def step(self, time_step):
        self.__val = np.sqrt(self.__scalar_strength) * time_step * randn(2) + self.__mean
        return
