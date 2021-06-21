# -*- coding: utf-8 -*-
from abc import ABC
import logging


class MeasurementModel(ABC):

    def __init__(self, measurement_noise, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self._noise = measurement_noise

    @property
    def noise(self):
        return self._noise.value

    def read(self, state):
        raise NotImplementedError('%s function not implemented' % self.read.__name__)

    def read_mean(self, state):
        raise NotImplementedError('%s function not implemented' % self.read.__name__)


class LinearMeasurementModel(MeasurementModel):

    def __init__(self, H, measurement_noise, logger=None):
        MeasurementModel.__init__(self, measurement_noise, logger)
        self.__H = H

    def read(self, state_vec):
        print(self._noise)
        return self.__H.dot(state_vec) + self._noise.step()

    def read_mean(self, state_vec):
        return self.__H.dot(state_vec)
