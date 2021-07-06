# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod  # abstract class
import numpy as np
from scipy.stats import multivariate_normal
import logging
from atomic_sensor_simulation.utilities import plot_data


class Noise(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self, num_steps):
        raise NotImplementedError


class GaussianWhiteNoise(Noise):
    """
     A class representing Wiener process: X(t + dt) = X(t) + N(0, Q * dt; t, t+dt), where Q is scalar strength.
     """

    def __init__(self, mean, cov, dt, logger=None):
        """
        :param cov:
        :param dt:
        :param mean:
        :param logger: instance of logging.Logger or None (if None a new instance of this class will be created)
        """
        self.__logger = logger if logger else logging.getLogger(__name__)
        self.__cov = cov
        self.__mean = mean
        self.__dt = dt
        self.__value = self.step()

    @property
    def value(self):
        return self.__value

    @property
    def cov(self):
        return self.__cov

    def step(self):
        self.__value = multivariate_normal.rvs(mean=self.__mean, cov=self.__cov)
        return self.__value

    def generate(self, num_steps):
        self.__logger.info('Generating Gaussian White Noise for %r steps' % str(num_steps))
        results = np.empty(num_steps)
        for x in range(num_steps):
            self.step()
            results[x] = self.__value
        times = np.arange(0, num_steps * self.__dt, self.__dt)
        return times, results

    def plot(self, num_steps=1000, output=None, is_show=False):
        self.__logger.info('Plotting Gaussian White Noise.')
        times, values = self.generate(num_steps)
        if output:
            plot_data(times,
                      values,
                      xlabel="time",
                      ylabel="Gaussian White Noise",
                      output=output,
                      is_show=is_show)
        else:
            plot_data(times,
                      values,
                      xlabel="time",
                      ylabel="Gaussian White Noise",
                      is_show=is_show)

    def cov_delta(self, delta_t):
        return self.__cov/delta_t
