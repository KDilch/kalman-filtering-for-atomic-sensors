# -*- coding: utf-8 -*-
import numpy as np
import logging
from enum import Enum

from atomic_sensor_simulation.dynamical_model.state import State

from atomic_sensor_simulation.operable_functions import create_operable_cos_func, create_operable_const_func, \
    create_operable_sin_func


class FrequencySensorCoordinates(Enum):
    """Enum translating vectors coordinates to human readable names."""
    X1 = 0
    X2 = 1
    X3 = 2


class FrequencySensorState(State):
    """
    Specialization of a dynamical_model abstract class. Represents a dynamical_model vector x_t_k = [x1,x2,x3].
    """

    def __init__(self, initial_vec, noise_vec, initial_time, time_arr, dt=1., logger=None, **kwargs):
        """
        :param initial_vec:
        :param noise_vec:
        :param initial_time: float; a member variable __time of class dynamical_model is initialized to initial time
        :param logger: an instance of logger.Logger; if not passed a new instance of a logger is initialized
        :param kwargs: key word args specific to a given simulation;
                       in this case they are: atoms_wiener_const, g_a_coupling_const
        """
        self.__time = initial_time
        self.slope = None
        self.shift = None
        State.__init__(self,
                       initial_vec,
                       noise_vec,
                       FrequencySensorCoordinates,
                       F_transition_matrix=None,
                       time_arr=time_arr,
                       x_jacobian=None,
                       dt=dt,
                       time=initial_time,
                       u_control_vec=np.array([create_operable_const_func(0.),
                                               create_operable_const_func(0.),
                                               create_operable_const_func(0.)]).T,
                       Gamma_control_evolution_matrix=np.array([[create_operable_const_func(1.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(0.)],
                                                                [create_operable_const_func(0.),
                                                                 create_operable_const_func(1.),
                                                                 create_operable_const_func(0.)],
                                                                [create_operable_const_func(0.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(1.)],
                                                                [create_operable_const_func(0.),
                                                                 create_operable_const_func(0.),
                                                                 create_operable_const_func(0.)]]),
                       logger=logger)

    @property
    def state_vec(self):
        """Returns a numpy array representing a dynamical_model vector x=(spin, quadrature)."""
        return self._state_vec

    @property
    def mean_state_vec(self):
        """Returns a numpy array representing a dynamical_model vector x without any noise."""
        return np.array(self._mean_state_vec)

    @property
    def noise_vec(self):
        """Returns a numpy array representing a noise vector [noise_j, noise_q]."""
        return self._noise_vec

    def linear_freq_change(self, t):
        if not self.slope:
            self.slope = 0.2
        if (t>self.time_arr[100]) and (t<self.time_arr[-100]):
            x = self.slope*t - self.slope*self.time_arr[100]
        elif t<self.time_arr[100]:
            x = 0.
        else:
            x=self.mean_state_vec[2]
        return x

    def step(self, t):
        self._time = t
        self._logger.debug('Performing a step for time %r' % str(self._time))
        index = np.where(self.time_arr == t)[0][0]
        self._mean_state_vec = np.array([np.cos(self.mean_state_vec[2]*self._dt)*self.mean_state_vec[0]-np.sin(self.mean_state_vec[2]*self._dt)*self.mean_state_vec[1],
                                         np.sin(self.mean_state_vec[2]*self._dt)*self.mean_state_vec[0]+np.cos(self.mean_state_vec[2]*self._dt)*self.mean_state_vec[1],
                                         self.mean_state_vec[2]])
        self._mean_state_vec[2] = self.linear_freq_change(t)
        self._state_vec = self._mean_state_vec + self.noise_vec.step()
        return
