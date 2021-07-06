# -*- coding: utf-8 -*-
from abc import ABC
import numpy as np
import logging


class MeasurementModel(ABC):

    def __init__(self, measurement_noise, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self._noise = measurement_noise

    @property
    def noise_val(self):
        return self._noise.value

    @property
    def noise_cov(self):
        return self._noise.cov

    def noise_cov_delta(self, delta):
        return self._noise.cov_delta(delta)

    def read(self, state):
        raise NotImplementedError('%s function not implemented' % self.read.__name__)

    def read_mean(self, state):
        raise NotImplementedError('%s function not implemented' % self.read.__name__)


class LinearMeasurementModel(MeasurementModel):

    def __init__(self, H, measurement_noise, logger=None):
        MeasurementModel.__init__(self, measurement_noise, logger)
        self.__H = H
        self.__H_inverse = np.linalg.pinv(H)

    def read(self, state_vec):
        return self.__H.dot(state_vec) + self._noise.step()

    def read_mean(self, state_vec):
        return self.__H.dot(state_vec)

    @property
    def H(self):
        return self.__H

    @property
    def H_inverse(self):
        return self.__H_inverse
