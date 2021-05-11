# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC
import logging
from scipy.signal import square, sawtooth
from atomic_sensor_simulation.utilities import eval_matrix_of_functions, operable
from atomic_sensor_simulation.operable_functions import create_operable_const_func

class LinearDynamics(object):
    "A class representing a state dynamics"
    def __init__(self,
                 F_transition_matrix,
                 time_step,
                 is_discrete,
                 time,
                 logger=None):
        self._logger = logger or logging.getLogger(__name__)
        self._F_transition_matrix = F_transition_matrix
        self._dt = time_step
        self._time = time
        self._is_discrete = is_discrete


class NonLinearDynamics(object):
    pass


class State(ABC):
    """An abstract class representing any dynamical_model vector."""

    def __init__(self,
                 initial_vec,
                 noise_vec,
                 coordinates_enum,
                 F_transition_matrix,
                 dt,
                 time,
                 time_arr,
                 gp=0,
                 omega_p=0,
                 logger=None):
        """
        :param initial_vec: numpy array
        :param noise_vec: numpy array
        :param coordinates_enum: indicates the order of coordinates using human readable names
        :param Phi_evolution_matrix: 
        :param u_control_vec:
        """
        self.gp = gp
        self.omega_p = omega_p
        self._logger = logger or logging.getLogger(__name__)
        self._state_vec = initial_vec
        self._mean_state_vec = initial_vec
        self._dt = dt
        self.time_arr = time_arr
        self.square_signal = square(2 * np.pi * time_arr / 6)
        self.sawtooth_signal = sawtooth(2 * np.pi * time_arr / 6)
        self.sin_signal = np.sin(2 * np.pi * time_arr / 6)
        self._noise_vec = noise_vec
        self._F_transition_matrix = F_transition_matrix
        self._x_jacobian = x_jacobian
        self._coordinates = coordinates_enum
        self._time = time
        self._waveform = None

    @property
    def state_vec(self):
        return self._state_vec

    @property
    def F_transition_matrix(self):
        return self._F_transition_matrix

    @property
    def x_Jacobian(self):
        return self._x_jacobian

    @property
    def mean_state_vec(self):
        return self._mean_state_vec


    @property
    def noise_vec(self):
        return self.noise_vec

    @property
    def time(self):
        return self._time

    @property
    def waveform(self):
        return self._waveform

    def step(self, t):
        self._logger.debug('Updating time and dt.')
        self._time = t
        self._logger.debug('Performing a step for time %r' % str(self._time))
        index = np.where(self.time_arr == t)[0][0]
        self._mean_state_vec = self._mean_state_vec + eval_matrix_of_functions(self._F_transition_matrix, t).dot(
            self.mean_state_vec) * self._dt
        # self._mean_state_vec[2] = self.square_signal[index]
        self._state_vec = self._mean_state_vec + self.noise_vec.step()
        self._waveform = self.gp * (
                    np.cos(self.omega_p * self._time) * self._state_vec[2] + np.sin(self.omega_p * self._time) *
                    self._state_vec[3])
        return
