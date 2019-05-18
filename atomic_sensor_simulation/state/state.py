# -*- coding: utf-8 -*-
import numpy as np
import logging
from abc import ABC, abstractmethod
from enum import Enum

from atomic_sensor_simulation.utilities import create_matrix_of_functions, exp_matrix_of_functions
from atomic_sensor_simulation.operable_functions import create_operable_sin_func, create_operable_cos_func, create_operable_const_func


class State(ABC):
    """An abstract class representing any state vector."""
    def __init__(self,
                 initial_vec,
                 noise_vec,
                 coordinates_enum,
                 F_evolution_matrix,
                 u_control_vec=None,
                 u_control_evolution_matrix=None):
        """
        :param initial_vec: numpy array
        :param noise_vec: numpy array
        :param coordinates_enum: indicates the order of coordinates using human readable names
        :param F_evolution_matrix: array of lambdas
        :param u_control_vec:
        """
        self._state_vec = initial_vec
        self._state_vec_no_noise = initial_vec
        self._noise_vec = noise_vec
        self._transition_matrix = F_evolution_matrix
        self._control_vec = u_control_vec
        self._control_evolution_matrix = u_control_evolution_matrix
        self._coordinates = coordinates_enum
        self._time = None

    @property
    def state_vec(self):
        return self._state_vec

    @property
    def F_evolution_matrix(self):
        return self._transition_matrix

    @property
    def u_control_vec(self):
        return self._control_vec

    @property
    def B_control_evolution_matrix(self):
        return self._control_evolution_matrix

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

    def __init__(self, initial_vec, noise_vec, initial_time, dt=1., logger=None, **kwargs):
        """
        :param initial_vec:
        :param noise_vec:
        :param initial_time: float; a member variable __time of class state is initialized to initial time
        :param logger: an instance of logger.Logger; if not passed a new instance of a logger is initialized
        :param kwargs: key word args specific to a given simulation;
                       in this case they are: atoms_wiener_const, g_a_coupling_const #TODO
        """
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a AtomicSensorState class.')
        self.__time = initial_time
        self.__atoms_wiener_correlation_const = kwargs['atoms_wiener_const']
        self.__g_a_coupling_const = kwargs['g_a_coupling_const']
        self.__control_amplitude = kwargs['control_amplitude']
        self.__control_freq = kwargs['control_freq']

        F_transition_matrix = exp_matrix_of_functions(create_matrix_of_functions(np.array(
                           [
                               [create_operable_const_func(self.__atoms_wiener_correlation_const*dt), create_operable_const_func(self.__g_a_coupling_const*dt)],
                               [create_operable_const_func(0), create_operable_const_func(dt)]
                           ])))

        State.__init__(self, initial_vec, noise_vec, AtomicSensorCoordinates,
                       F_evolution_matrix=F_transition_matrix,
                       u_control_vec=create_matrix_of_functions(np.array([create_operable_const_func(0), create_operable_const_func(1)]).T),
                       u_control_evolution_matrix=create_matrix_of_functions(np.array([[create_operable_const_func(0), create_operable_const_func(0)],
                                                                          [create_operable_const_func(0), create_operable_cos_func(self.__control_amplitude, self.__control_freq)]])))

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

    @property
    def time(self):
        return self._time

    def __noise_step(self):
        noise_val_vec = np.zeros(len(self.noise_vec))
        for n in range(len(self.noise_vec)):
            self.noise_vec[n].step()
            noise_val_vec[n] = self.noise_vec[n].value
            print(noise_val_vec)
        return np.array(noise_val_vec)

    def step(self, t):
        self.__logger.debug('Updating time and dt.')
        self._time = t
        self.__logger.debug('Performing a step for time %r' % str(self._time))
        F = self._transition_matrix(self._time)
        u = self._control_vec(self._time)
        B = self._control_evolution_matrix(self._time)
        from atomic_sensor_simulation import CONSTANTS
        Q = F.dot(np.eye(2)).dot(np.array([[(CONSTANTS.SCALAR_STREGTH_j), (0)], [(0), (CONSTANTS.SCALAR_STRENGTH_q)]])).dot(
            np.eye(2).T).dot(F.T)

        self._state_vec = F.dot(self.state_vec_no_noise) +\
                          self.__noise_step() +\
                          0
        self._state_vec_no_noise = F.dot(self.state_vec_no_noise)
        return

class PositionVelocitySensorState(object):
    pass