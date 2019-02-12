# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod  # abstract class
import numpy as np
from scipy.stats import norm
import logging
from atomic_sensor_simulation.utilities import plot_data


class Noise(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def values(self):
        raise NotImplementedError


class GaussianWhiteNoise(Noise):
    """
     A class representing Wiener process: X(t + dt) = X(t) + N(0, Q * dt; t, t+dt), where Q is scalar strength.
     """

    def __init__(self, initial_value, scalar_strength, dt, num_steps=1000, mean=0, logger=None):
        """
        :param initial_value: float
        :param scalar_strength: float
        :param dt: float
        :param num_steps: int, default 1000
        :param mean: float, default 0
        :param logger: instance of logging.Logger or None (if None a new instance of this class will be created)
        """
        self.__logger = logger if logger else logging.getLogger(__name__)
        self.__initial_state = initial_value
        self.__current_value = initial_value
        self.__scalar_strength = scalar_strength
        self.__mean = mean
        self.__dt = dt
        self.__num_steps = num_steps
        self.__values = None
        self.__times = np.arange(0, self.__num_steps*self.__dt, self.__dt)
        self.__generate()

    @property
    def values(self):
        """
        :return: tuple with 2 numpy arrays (time, noise)
        """
        return self.__times, self.__values

    def step(self):
        self.__current_value = norm.rvs(loc=self.__mean, size=1, scale=np.sqrt(self.__scalar_strength * self.__dt))[0]
        return

    def __generate(self):
        self.__logger.info('Generating Gaussian White Noise for %r steps' % str(self.__num_steps))
        results = np.empty(self.__num_steps)
        for x in range(self.__num_steps):
            self.step()
            results[x] = self.__current_value
        results += np.expand_dims(self.__initial_state, axis=-1)
        self.__values = results
        return

    def plot(self, output=None, is_show=False):
        self.__logger.info('Plotting Gaussian White Noise.')
        if output:
            plot_data(self.__times,
                      self.__values,
                      xlabel="time",
                      ylabel="Gaussian White Noise",
                      output=output,
                      is_show=is_show)
        else:
            plot_data(self.__times,
                      self.__values,
                      xlabel="time",
                      ylabel="Gaussian White Noise",
                      is_show=is_show)
