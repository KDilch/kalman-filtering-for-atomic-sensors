# -*- coding: utf-8 -*-
import numpy as np
import logging
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class Spin(object):

    def __init__(self, larmor_freq, t2_param, scalar_strength, dt, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a Spin class.')
        self.__noise = GaussianWhiteNoise(scalar_strength, dt=dt, scalar_strength=scalar_strength)
        self.__larmor_freq = larmor_freq
        self.__t2_param = t2_param
        self.__dt = dt
        self.__evolution_matrix = np.array([[-1. / self.__t2_param, self.__larmor_freq], [-self.__larmor_freq, -1. / self.__t2_param]])
        self.__val = np.empty(2)
        self.__val_previous = np.empty(2)

    @property
    def val(self):
        return self.__val

    @property
    def t2_param(self):
        return self.__t2_param

    @property
    def larmor_freq(self):
        return self.__larmor_freq

    @property
    def evolution_matrix(self):
        return self.__evolution_matrix

    def step(self):
        assert isinstance(self.__val, type(np.array([])))
        self.__val_previous = self.__val
        self.__val = np.matmul(self.__evolution_matrix, self.__val * self.__dt)


class Quadrature(object):
    """Evolution is Ornstein Uhlenbeck Process"""

    def __init__(self, p, q, correlation_time, scalar_strength, dt, initial_time=0, logger=None):
        """
        :param p: function or lambda
        :param q: function or lambda
        """
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a Quadrature class.')
        self.__p = p
        self.__q = q
        self.__val = np.array([self.__p, self.__q])  # for now p,q const
        self.__val_previous = np.empty(2)
        self.__correlation_time = correlation_time
        self.__noise = GaussianWhiteNoise(scalar_strength, dt, scalar_strength)
        self.__dt = dt
        self.__evolution_matrix = np.array([[-self.__correlation_time, 0], [0, -self.__correlation_time]])

    @property
    def evolution_matrix(self):
        return self.__evolution_matrix

    @property
    def val(self):
        return self.__val

    def step(self):
        assert isinstance(self.__val, type(np.array([])))
        self.__val_previous = self.__val
        self.__val = self.__evolution_matrix.dot(self.__val) * self.__dt


class Signal(object):
    """Circularly polarized signal"""

    def __init__(self, signal_freq, quadrature, coupling_const, dt, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a IcfoSignal class.')
        self.__signal_freq = signal_freq
        self.__quadrature = quadrature
        self.__coupling_const = coupling_const
        self.__evolution_matrix = np.array([[0, 0], [self.__coupling_const*np.sin(self.__signal_freq), self.__coupling_const*np.cos(self.__signal_freq)]])
        self.__val = None
        self.__time = 0
        self.__dt = dt
        self.__noise = None

    @property
    def val(self):
        if self.__val is not None:
            return self.__val
        else:
            Warning('Not current value defined for signal. You need to step first.')

    @property
    def quadrature(self):
        return self.__quadrature

    @property
    def evolution_matrix(self):
        return self.__evolution_matrix

    def step(self):
        self.__time += self.__dt
        self.__quadrature.step()
        self.__val = self.__func(self.__time).dot(self.__quadrature.val)
        return

    def __func(self, time):
        return np.array([self.__coupling_const*np.cos(self.__signal_freq*time), self.__coupling_const*np.sin(self.__signal_freq*time)])


class State(object):
    """
    Represents a state vector x_t_k = [j_y, j_z, q(t_k), p(t_k)].
    Implement initialization step in here. So that we can feed KBF with a proper value. Use the numbers provided in the paper.
    Plot state without noise. Plot state with Gaussian noise. Show what KBF does.
    """

    def __init__(self, spin, signal, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a State class.')
        self.__spin = spin
        self.__signal = signal
        self.__val = np.array([spin, signal.quadrature])
        self.__evolution_matrix = np.vstack((np.hstack((spin.evolution_matrix, signal.evolution_matrix)),
                                             np.hstack((np.array([[0, 0], [0, 0]]), signal.quadrature.evolution_matrix))
                                             )
                                            )

    @property
    def evolution_matrix(self):
        return self.__evolution_matrix

    @property
    def val(self):
        return self.__val

    @property
    def signal(self):
        return self.__signal.val

    def step(self):
        """Making a step."""
        self.__signal.step()
        self.__spin.step()
        self.__val = np.array([self.__spin.val[0], self.__signal.quadrature.val])

    def get_sensor_reading(self):
        """In this particular experiment photocurrent is measured. Think if this function should be a part of this class."""
        pass
