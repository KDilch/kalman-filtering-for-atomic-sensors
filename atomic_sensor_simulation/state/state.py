# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC, abstractmethod


class State(ABC):
    """An abstract class representing any state vector."""
    def __init__(self,
                 initial_vec,
                 noise_vec,
                 coordinates_enum,
                 F_transition_matrix,
                 u_control_vec=None,
                 Gamma_control_evolution_matrix=None,
                 initial_control_vec=None):
        """
        :param initial_vec: numpy array
        :param noise_vec: numpy array
        :param coordinates_enum: indicates the order of coordinates using human readable names
        :param Phi_evolution_matrix: 
        :param u_control_vec:
        """
        self._state_vec = initial_vec
        self._mean_state_vec = initial_vec
        if initial_control_vec:
            self._control_state_vec = initial_control_vec
        else:
            self._control_state_vec = np.zeros(len(self._state_vec))
        self._noise_vec = noise_vec
        self._F_transition_matrix = F_transition_matrix
        self._u_control_vec = u_control_vec
        self._Gamma_control_evolution_matrix = Gamma_control_evolution_matrix
        self._coordinates = coordinates_enum
        self._time = None

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
    @abstractmethod
    def time(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, time):
        raise NotImplementedError
