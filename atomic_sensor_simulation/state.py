# -*- coding: utf-8 -*-
import numpy as np
import logging
from abc import ABC, abstractmethod
from enum import Enum


class State(ABC):
    """An abstract class representing any state vector."""
    def __init__(self, initial_vec, noise_vec, coordinates_enum, evolution_matrix, control_func=None):
        """
        :param initial_vec: numpy array
        :param noise_vec: numpy array
        :param coordinates_enum: indicates the order of coordinates using human readable names
        :param evolution_matrix: function that needs to be evaluated for a number of params and spits out a numpy array
        :param control_func:
        """
        self._state_vec = initial_vec
        self._state_vec_no_noise = initial_vec
        self._noise_vec = noise_vec
        self._evolution_matrix = evolution_matrix
        self._control_func_vec = control_func
        self._coordinates = coordinates_enum
        self._time = None  # keep current time in memory for which the given state and noise vecs were computed

    @property
    def state_vec(self):
        return self._state_vec

    @property
    def state_vec_no_noise(self):
        return self._state_vec_no_noise

    @property
    def noise_vec(self):
        return self.noise_vec

    @property
    @abstractmethod
    def time(self):
        raise NotImplementedError

    @abstractmethod
    def evolution_matrix(self, t):
        raise NotImplementedError

    @abstractmethod
    def step(self, time):
        raise NotImplementedError


class AtomicSensorCoordinates(Enum):
    """Enum translating vectors coordinates to human readable names."""
    SPIN = 0
    QUADRATURE = 1


class AtomicSensorState(State):
    """
    Specialization of a State abstract class. Represents a state vector x_t_k = [j, q].
    """

    def __init__(self, initial_vec, noise_vec, initial_time, logger=None, **kwargs):
        """
        :param initial_vec:
        :param noise_vec:
        :param initial_time: float; a member variable __time of class state is initialized to initial time
        :param logger: an instance of logger.Logger; if not passed a new instance of a logger is initialized
        :param kwargs: key word args specific to a given simulation;
                       in this case they are: atoms_wiener_const, g_a_coupling_const
        """
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a AtomicSensorState class.')
        State.__init__(self, initial_vec, noise_vec, AtomicSensorCoordinates, evolution_matrix=self.evolution_matrix, control_func=None)
        self.__time = initial_time
        self.__atoms_wiener_correlation_const = kwargs['atoms_wiener_const']#availble params are: coupling const, omega, amplidude of the signal etc.#TODO
        self.__g_a_coupling_const = kwargs['g_a_coupling_const']

    @property
    def state_vec(self):
        """Returns a numpy array representing a state vector x=(spin, quadrature)."""
        return self._state_vec

    @property
    def state_vec_no_noise(self):
        """Returns a numpy array representing a state vector x without any noise."""
        return self._state_vec_no_noise

    @property
    def noise_vec(self):
        """Returns a numpy array representing a noise vector [noise_j, noise_q]."""
        return self._noise_vec

    @property
    def spin(self):
        return self._state_vec[self._coordinates.SPIN.value]

    @property
    def quadrature(self):
        return self._state_vec[self._coordinates.QUADRATURE.value]

    @property
    def spin_no_noise(self):
        return self._state_vec_no_noise[self._coordinates.SPIN.value]

    @property
    def quadrature_no_noise(self):
        return self._state_vec_no_noise[self._coordinates.SPIN.value]
        # return self.__amplitude*np.sin(self.__omega*t) TODO make a quadrature a function

    @property
    def time(self):
        return self._time

    def __noise_step(self):
        noise_val_vec = np.zeros(len(self.noise_vec))
        for n in range(len(self.noise_vec)):
            self.noise_vec[n].step()
            noise_val_vec[n] = self.noise_vec[n].value
        return noise_val_vec

    def evolution_matrix(self, t):
        """Evolution matrix evaluated for t and dt."""
        return np.array([[self.__atoms_wiener_correlation_const, self.__g_a_coupling_const],
                        [0, 1]])

    def step(self, t):
        self.__logger.debug('Updating time and dt.')
        self._time = t
        self.__logger.debug('Performing a step for time %r' % str(self._time))

        self._state_vec = self.evolution_matrix(self._time).dot(self.state_vec_no_noise) + self.__noise_step()
        self._state_vec_no_noise = self.evolution_matrix(self._time).dot(self.state_vec_no_noise)
        return

        # self.__spin = -self.__atom_correlation_conts*self.__spin_no_noise*self.__dt + g_a_COUPLING_CONST * self.quadrature_no_noise(t) * self.__dt + self.__noise[0].step()
        # self.__quadrature = self.quadrature_no_noise(t) + self.__noise[1].step() + self.__amplitude*np.sin(self.__omega*t)
        # self.__spin_no_noise = -self.__atom_correlation_conts*self.__spin_no_noise*self.__dt + g_a_COUPLING_CONST * self.quadrature_no_noise(t)
