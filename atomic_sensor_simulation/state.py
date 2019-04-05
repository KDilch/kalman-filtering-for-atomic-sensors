# -*- coding: utf-8 -*-
import numpy as np
import logging
from atomic_sensor_simulation.CONSTANTS import g_a_COUPLING_CONST


class State(object):
    """
    Represents a state vector x_t_k = [j, q]. Only Wiener Processes.
    """

    def __init__(self, spin, quadrature, noise_spin, noise_quadrature, dt, logger=None, atoms_correlation_const = 0, omega=0., amplitude=0.):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a State class.')
        self.__atom_correlation_conts = atoms_correlation_const
        self.__omega = omega
        self.__spin = spin
        self.__quadrature = quadrature
        self.__spin_no_noise = spin
        self.__quadrature_no_noise = quadrature
        self.__noise = np.array([noise_spin, noise_quadrature])
        self.__dt = dt
        self.__evolution_matrix = np.array([[1, 0], [g_a_COUPLING_CONST * self.__dt, 1]])
        self.__amplitude = amplitude

    @property
    def evolution_matrix(self):
        return self.__evolution_matrix

    @property
    def vec(self):
        """Returns a state vector x=(spin, quadrature)."""
        return np.array([self.__spin, self.__quadrature])

    @property
    def vec_no_noise(self):
        """Returns a state vector x without any noise."""
        return np.array([self.__spin_no_noise, self.__quadrature_no_noise])

    @property
    def spin(self):
        return self.__spin

    @property
    def quadrature(self):
        return self.__quadrature

    @property
    def spin_no_noise(self):
        return self.__spin_no_noise

    def quadrature_no_noise(self, t):
        return self.__quadrature_no_noise
        # return self.__amplitude*np.sin(self.__omega*t)

    @property
    def noise(self):
        return self.__noise

    def step(self, t):
        self.__logger.debug('Executing step function.')
        self.__spin = -self.__atom_correlation_conts*self.__spin_no_noise*self.__dt + g_a_COUPLING_CONST * self.quadrature_no_noise(t) * self.__dt + self.__noise[0].step()
        self.__quadrature = self.quadrature_no_noise(t) + self.__noise[1].step() + self.__amplitude*np.sin(self.__omega*t)
        self.__spin_no_noise = -self.__atom_correlation_conts*self.__spin_no_noise*self.__dt + g_a_COUPLING_CONST * self.quadrature_no_noise(t)
        return
