# -*- coding: utf-8 -*-
import numpy as np
import logging
from enum import Enum

from atomic_sensor_simulation.state.state import State

from atomic_sensor_simulation.operable_functions import create_operable_cos_func, create_operable_const_func, \
    create_operable_sin_func


class AtomicSensorCoordinates(Enum):
    """Enum translating vectors coordinates to human readable names."""
    SPIN_Y = 0
    SPIN_Z = 1
    QUADRATURE_P = 2
    QUADRATURE_Q = 3


class AtomicSensorState(State):
    """
    Specialization of a state abstract class. Represents a state vector x_t_k = [j, q].
    """

    def __init__(self, initial_vec, noise_vec, initial_time, time_arr, dt=1., logger=None, **kwargs):
        """
        :param initial_vec:
        :param noise_vec:
        :param initial_time: float; a member variable __time of class state is initialized to initial time
        :param logger: an instance of logger.Logger; if not passed a new instance of a logger is initialized
        :param kwargs: key word args specific to a given simulation;
                       in this case they are: atoms_wiener_const, g_a_coupling_const
        """
        self.__time = initial_time
        self.__light_correlation_const = kwargs['light_correlation_const']
        self.__coupling_amplitude = kwargs['coupling_amplitude']
        self.__coupling_freq = kwargs['coupling_freq']
        self.__coupling_phase_shift = kwargs['coupling_phase_shift']
        self.__larmour_freq = kwargs['larmour_freq']
        self.__spin_correlation_const = kwargs['spin_correlation_const']

        # F is a matrix with entries being functions. This wa can be integrated easily. Keep funcitons defined in "operable_functions"
        F_transition_matrix = np.array([[create_operable_const_func(-self.__spin_correlation_const),
                                         create_operable_const_func(self.__larmour_freq),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0)],

                                        [create_operable_const_func(-self.__larmour_freq),
                                         create_operable_const_func(-self.__spin_correlation_const),
                                         create_operable_cos_func(amplitude=self.__coupling_amplitude,
                                                                  omega=self.__coupling_freq,
                                                                  phase_shift=self.__coupling_phase_shift),
                                         create_operable_sin_func(amplitude=self.__coupling_amplitude,
                                                                  omega=self.__coupling_freq,
                                                                  phase_shift=self.__coupling_phase_shift),
                                         create_operable_const_func(0)],

                                        [create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(-self.__light_correlation_const),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0)],

                                        [create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(-self.__light_correlation_const),
                                         create_operable_const_func(0)],
                                        [create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(1)]])

        x_JACOBIAN = np.array([[create_operable_const_func(1-self.__spin_correlation_const),
                                create_operable_const_func(self.__larmour_freq),
                                create_operable_const_func(0),
                                create_operable_const_func(0),
                                create_operable_const_func(0)],

                               [create_operable_const_func(-self.__larmour_freq),
                                create_operable_const_func(1-self.__spin_correlation_const),
                                create_operable_cos_func(amplitude=-self.__coupling_amplitude * self.__coupling_freq,
                                                         omega=self.__coupling_freq,
                                                         phase_shift=self.__coupling_phase_shift),
                                create_operable_sin_func(amplitude=self.__coupling_amplitude + self.__coupling_freq,
                                                         omega=self.__coupling_freq,
                                                         phase_shift=self.__coupling_phase_shift),
                                create_operable_const_func(0)],

                               [create_operable_const_func(0),
                                create_operable_const_func(0),
                                create_operable_const_func(1-self.__light_correlation_const),
                                create_operable_const_func(0),
                                create_operable_const_func(0)
                                ],

                               [create_operable_const_func(0),
                                create_operable_const_func(0),
                                create_operable_const_func(0),
                                create_operable_const_func(1-self.__light_correlation_const),
                                create_operable_const_func(0)]])

        State.__init__(self,
                       initial_vec,
                       noise_vec,
                       AtomicSensorCoordinates,
                       F_transition_matrix=F_transition_matrix,
                       time_arr=time_arr,
                       x_jacobian=x_JACOBIAN,
                       dt=dt,
                       time=initial_time,
                       gp=self.__coupling_amplitude,
                       omega_p=self.__coupling_freq,
                       u_control_vec=np.array([create_operable_const_func(0.),
                                               create_operable_const_func(0.),
                                               create_operable_const_func(0.),
                                               create_operable_const_func(0.)]).T,
                       Gamma_control_evolution_matrix=np.array([[create_operable_const_func(1.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(0.)],
                                                                [create_operable_const_func(0.),
                                                                 create_operable_const_func(1.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(0.)],
                                                                [create_operable_const_func(0.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(1.),
                                                                 create_operable_const_func(0.)],
                                                                [create_operable_const_func(0.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(1.)]]),
                       logger=logger)

    @property
    def state_vec(self):
        """Returns a numpy array representing a state vector x=(spin, quadrature)."""
        return self._state_vec

    @property
    def mean_state_vec(self):
        """Returns a numpy array representing a state vector x without any noise."""
        return np.array(self._mean_state_vec)

    @property
    def noise_vec(self):
        """Returns a numpy array representing a noise vector [noise_j, noise_q]."""
        return self._noise_vec

    @property
    def spin(self):
        return [self._state_vec[self._coordinates.SPIN_Y.value], self._state_vec[self._coordinates.SPIN_Z.value]]

    @property
    def quadrature(self):
        return [self._state_vec[self._coordinates.QUADRATURE_Q.value],
                self._state_vec[self._coordinates.QUADRATURE_P.value]]

    @property
    def spin_mean(self):
        return [self._mean_state_vec[self._coordinates.SPIN_Y.value],
                self._mean_state_vec[self._coordinates.SPIN_Z.value]]

    @property
    def quadrature_mean(self):
        return [self._mean_state_vec[self._coordinates.QUADRATURE_Q.value],
                self._mean_state_vec[self._coordinates.QUADRATURE_P.value]]

