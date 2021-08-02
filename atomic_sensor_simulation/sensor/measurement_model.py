# -*- coding: utf-8 -*-
from abc import ABC
import numpy as np
import logging
from functools import cached_property

class MeasurementModel(ABC):

    def __init__(self, measurement_noise, dt, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self._noise = measurement_noise
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    @property
    def noise_val(self):
        return self._noise.value

    @property
    def noise_cov(self):
        return self._noise.cov

    @cached_property
    def noise_cov_delta(self):
        return self._noise.cov_delta(self._dt)

    def read(self, state):
        raise NotImplementedError('%s function not implemented' % self.read.__name__)

    def read_mean(self, state):
        raise NotImplementedError('%s function not implemented' % self.read.__name__)


class LinearMeasurementModel(MeasurementModel):

    def __init__(self, H, measurement_noise, dt, logger=None):
        MeasurementModel.__init__(self, measurement_noise, dt, logger)
        self.__H = H

    def read(self, state_vec):
        return self.__H.dot(state_vec) + self._noise.step()

    def read_mean(self, state_vec):
        return self.__H.dot(state_vec)

    @property
    def H(self):
        return self.__H

    @property
    def H_T(self):
        return np.transpose(self.__H)

    @property
    def H_inverse(self):
        return np.linalg.pinv(self.__H)
