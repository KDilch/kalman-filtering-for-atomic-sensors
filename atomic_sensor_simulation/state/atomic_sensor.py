# -*- coding: utf-8 -*-
import numpy as np
import logging
from enum import Enum

from atomic_sensor_simulation.state.state import State
from atomic_sensor_simulation.utilities import eval_matrix_of_functions

from atomic_sensor_simulation.operable_functions import create_operable_cos_func, create_operable_const_func


class AtomicSensorCoordinates(Enum):
    """Enum translating vectors coordinates to human readable names."""
    SPIN = 0
    QUADRATURE = 1


class AtomicSensorState(State):
    """
    Specialization of a state abstract class. Represents a state vector x_t_k = [j, q].
    """

    def __init__(self, initial_vec, noise_vec, initial_time, dt=1., logger=None, **kwargs):
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
        self.__time = initial_time
        self.__atoms_wiener_correlation_const = kwargs['atoms_wiener_const']
        self.__g_a_coupling_const = kwargs['g_a_coupling_const']
        self.__control_amplitude = kwargs['control_amplitude']
        self.__control_freq = kwargs['control_freq']
        self.__spin_correlation_const = kwargs['spin_correlation_const']
        self.__dt = dt
        F_transition_matrix = np.array([[create_operable_const_func(-self.__spin_correlation_const), create_operable_cos_func(amplitude=self.__control_amplitude, omega=self.__control_freq, phase_shift=5.)],
                                        [create_operable_const_func(0), create_operable_const_func(-self.__atoms_wiener_correlation_const)]])

        State.__init__(self,
                       initial_vec,
                       noise_vec,
                       AtomicSensorCoordinates,
                       F_transition_matrix=F_transition_matrix,
                       u_control_vec=np.array([create_operable_const_func(0.), create_operable_const_func(0.)]).T,
                       Gamma_control_evolution_matrix=np.array([[create_operable_const_func(1.), create_operable_const_func(0.)],
                                                                [create_operable_const_func(0.), create_operable_const_func(1.)]]))

    @property
    def state_vec(self):
        """Returns a numpy array representing a state vector x=(spin, quadrature)."""
        return self._state_vec

    @property
    def mean_state_vec(self):
        """Returns a numpy array representing a state vector x without any noise."""
        return self._mean_state_vec

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
    def spin_mean(self):
        return self._mean_state_vec[self._coordinates.SPIN.value]

    @property
    def quadrature_mean(self):
        return self._mean_state_vec[self._coordinates.QUADRATURE.value]

    @property
    def time(self):
        return self._time

    def __noise_step(self):
        noise_val_vec = np.zeros(len(self.noise_vec))
        for n in range(len(self.noise_vec)):
            self.noise_vec[n].step()
            noise_val_vec[n] = self.noise_vec[n].value
        return np.array(noise_val_vec)

    def step(self, t):
        self.__logger.debug('Updating time and dt.')
        self._time = t
        self.__logger.debug('Performing a step for time %r' % str(self._time))
        self._mean_state_vec = self._mean_state_vec + eval_matrix_of_functions(self._F_transition_matrix, t).dot(self.mean_state_vec) * self.__dt
        # self._control_state_vec = self._control_state_vec + eval_matrix_of_functions(self._Gamma_control_evolution_matrix, t).dot(eval_matrix_of_functions(self._u_control_vec, t))*self.quadrature * self.__dt
        self._state_vec = self._mean_state_vec + self.__noise_step()
        return
