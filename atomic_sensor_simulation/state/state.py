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
                 Phi_evolution_matrix,
                 u_control_vec=None,
                 u_control_evolution_matrix=None):
        """
        :param initial_vec: numpy array
        :param noise_vec: numpy array
        :param coordinates_enum: indicates the order of coordinates using human readable names
        :param Phi_evolution_matrix: 
        :param u_control_vec:
        """
        self._state_vec = initial_vec
        self._state_vec_no_noise = initial_vec
        self._noise_vec = noise_vec
        self._Phi_evolution_matrix = Phi_evolution_matrix
        self._control_vec = u_control_vec
        self._control_evolution_matrix = u_control_evolution_matrix
        self._coordinates = coordinates_enum
        self._time = None

    @property
    def state_vec(self):
        return self._state_vec

    @property
    def Phi_evolution_matrix(self):
        return self._Phi_evolution_matrix

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
