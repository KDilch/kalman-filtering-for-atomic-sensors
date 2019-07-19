# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC
import logging

from atomic_sensor_simulation.utilities import eval_matrix_of_functions, operable
from atomic_sensor_simulation.operable_functions import create_operable_const_func

class State(ABC):
    """An abstract class representing any state vector."""
    def __init__(self,
                 initial_vec,
                 noise_vec,
                 coordinates_enum,
                 F_transition_matrix,
                 dt,
                 time,
                 u_control_vec=None,
                 Gamma_control_evolution_matrix=None,
                 initial_control_vec=None,
                 logger=None):
        """
        :param initial_vec: numpy array
        :param noise_vec: numpy array
        :param coordinates_enum: indicates the order of coordinates using human readable names
        :param Phi_evolution_matrix: 
        :param u_control_vec:
        """
        self._logger = logger or logging.getLogger(__name__)
        self._state_vec = initial_vec
        self._mean_state_vec = initial_vec
        self._dt = dt
        if initial_control_vec:
            self._control_state_vec = initial_control_vec
        else:
            self._control_state_vec = np.zeros_like(self._state_vec)
        self._noise_vec = noise_vec
        self._F_transition_matrix = F_transition_matrix
        if u_control_vec is not None:
            self._u_control_vec = u_control_vec
        else:
            self._u_control_vec = np.array([create_operable_const_func(0.)for j in range(np.shape(self._state_vec)[0])])
        if Gamma_control_evolution_matrix is not None:
            self._Gamma_control_evolution_matrix = Gamma_control_evolution_matrix
        else:
            self._Gamma_control_evolution_matrix = np.array([[create_operable_const_func(0.) for i in range(np.shape(self._F_transition_matrix)[0])] for j in range(np.shape(self._F_transition_matrix)[0])])
        self._coordinates = coordinates_enum
        self._time = time

    @property
    def state_vec(self):
        return self._state_vec

    @property
    def F_transition_matrix(self):
        return self._F_transition_matrix

    @property
    def u_control_vec(self):
        return self._u_control_vec

    @property
    def Gamma_control_evolution_matrix(self):
        return self._Gamma_control_evolution_matrix

    @property
    def mean_state_vec(self):
        return self._mean_state_vec

    @property
    def control_state_vec(self):
        return self._control_state_vec

    @property
    def noise_vec(self):
        return self.noise_vec

    @property
    def time(self):
        return self._time

    def step(self, t):
        self._logger.debug('Updating time and dt.')
        self._time = t
        self._logger.debug('Performing a step for time %r' % str(self._time))
        self._mean_state_vec = self._mean_state_vec + eval_matrix_of_functions(self._F_transition_matrix, t).dot(self.mean_state_vec) * self._dt
        self._state_vec = self._mean_state_vec + self.noise_vec.step()
        # Incorporating u(t) into the dynamics
        #self._control_state_vec = self._control_state_vec + eval_matrix_of_functions(self._Gamma_control_evolution_matrix, t).dot(eval_matrix_of_functions(self._u_control_vec, t)) * self._dt
        # self._state_vec = self._mean_state_vec + self._control_state_vec + self._noise_step()
        return

