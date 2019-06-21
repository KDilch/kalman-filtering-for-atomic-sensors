# -*- coding: utf-8 -*-
import numpy as np
import logging
from enum import Enum

from atomic_sensor_simulation.state.state import State
from atomic_sensor_simulation.utilities import create_matrix_of_functions, exp_matrix_of_functions
from atomic_sensor_simulation.operable_functions import create_operable_cos_func, create_operable_const_func


class PosVelSensorCoordinates(Enum):
    """Enum translating vectors coordinates to human readable names."""
    POSITION_X = 0
    VELOCITY_X = 1
    POSITION_Y = 2
    VELOCITY_Y = 3


class PosVelSensorState(State):
    """
    Specialization of a state abstract class. Represents a state vector x_t_k = [x, v].
    Each of the coordinates is a 2-dim vector.
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
        self.__logger.info('Initializing an instance of a PosVelSensor class.')
        self.__time = initial_time

        F_transition_matrix = create_matrix_of_functions(np.array(
                           [
                               [create_operable_const_func(1.),
                                create_operable_const_func(dt),
                                create_operable_const_func(0.),
                                create_operable_const_func(0.)],

                               [create_operable_const_func(0.),
                                create_operable_const_func(1.),
                                create_operable_const_func(0.),
                                create_operable_const_func(0.)],

                               [create_operable_const_func(0.),
                                create_operable_const_func(0.),
                                create_operable_const_func(1.),
                                create_operable_const_func(dt)],

                               [create_operable_const_func(0.),
                                create_operable_const_func(0.),
                                create_operable_const_func(0.),
                                create_operable_const_func(1.)]
                           ]))

        State.__init__(self, initial_vec, noise_vec, PosVelSensorCoordinates,
                       Phi_evolution_matrix=F_transition_matrix,
                       u_control_vec=0,
                       u_control_evolution_matrix=0)

    @property
    def state_vec(self):
        """Returns a numpy array representing a state vector x=(spin, quadrature)."""
        return self._state_vec

    @property
    def mean_state_vec(self):
        """Returns a numpy array representing a state vector x without any noise."""
        return self._state_vec_no_noise

    @property
    def noise_vec(self):
        """Returns a numpy array representing a noise vector [noise_j, noise_q]."""
        return self._noise_vec

    @property
    def position_x(self):
        return self._state_vec[self._coordinates.POSITION_X.value]

    @property
    def position_y(self):
        return self._state_vec[self._coordinates.POSITION_Y.value]

    @property
    def velocity_x(self):
        return self._state_vec[self._coordinates.VELOCITY_X.value]

    @property
    def velocity_y(self):
        return self._state_vec[self._coordinates.VELOCITY_Y.value]

    @property
    def position_x_no_noise(self):
        return self._state_vec[self._coordinates.POSITION_X.value]

    @property
    def position_y_no_noise(self):
        return self._state_vec[self._coordinates.POSITION_Y.value]

    @property
    def velocity_x_no_noise(self):
        return self._state_vec[self._coordinates.VELOCITY_X.value]

    @property
    def velocity_y_no_noise(self):
        return self._state_vec[self._coordinates.VELOCITY_Y.value]

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
        F = self._Phi_evolution_matrix(self._time)
        # u = self._control_vec(self._time)
        # B = self._control_evolution_matrix(self._time)
        from numpy.random import randn

        self._state_vec = F.dot(self.mean_state_vec) + \
                          np.array([randn() *1., randn() * 1., 0., 0.])
        self._state_vec_no_noise = F.dot(self.mean_state_vec)
        return