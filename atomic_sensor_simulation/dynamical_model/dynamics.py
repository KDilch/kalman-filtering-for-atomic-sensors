# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC
import logging
from scipy.linalg import expm
from scipy.integrate import odeint, simps


class DynamicalModel(ABC):
    """Class holding a matrix of functions - linear/ non-linear, discrete/continuous, time-dependent/time independent"""

    def __init__(self,
                 transition_matrix,
                 dt,
                 intrinsic_noise=None,
                 is_model_differential=True,
                 discretization_active=False,
                 is_model_time_invariant=False,
                 discrete_dt=None
                 ):
        """
        :param transition_matrix: matrix of functions
        """
        self.transition_matrix = transition_matrix
        self.dt = dt
        self.intrinsic_noise = intrinsic_noise
        self.__state_vec_shape = transition_matrix.shape[0]
        self.__is_model_differential = is_model_differential
        self.__discretization_active = discretization_active
        self.__is_model_time_invariant = is_model_time_invariant
        if self.__is_model_differential and self.__discretization_active:
            self.discrete_transition_matrix = None
            self.discrete_transition_matrix_T = None
            self.discrete_intrinsic_noise = None
            self.discrete_dt = discrete_dt

    def evaluate_transition_matrix_at_time_t(self, time):
        """Evaluates the matrix of functions at time t"""
        matrix_flat = self.transition_matrix.flatten()
        shape = np.shape(self.transition_matrix)
        evaluated_matrix = np.empty_like(matrix_flat)
        for index, element in np.ndenumerate(matrix_flat):
            evaluated_matrix[index] = matrix_flat[index](time)
        return np.reshape(evaluated_matrix, shape)

    def num_compute_discrete_transition_noise_matrices(self, from_time, to_time, time_resolution):
        """Computes discrete model (Phi, Q^{\Delta})"""
        if self.__is_model_differential and self.__discretization_active:
            Phi_0 = np.reshape(np.identity(self.__state_vec_shape),
                               self.__state_vec_shape ** 2)  # transition matrix from_time to from_time is identity

            def dPhidt(Phi, t):
                return np.reshape(np.dot(np.array(self.evaluate_transition_matrix_at_time_t(time=t), dtype=float),
                                         np.reshape(Phi, (self.__state_vec_shape, self.__state_vec_shape))),
                                  self.__state_vec_shape ** 2)

            t = np.linspace(from_time, to_time, num=time_resolution)  # times to report solution
            # solve ODE
            Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, self.__state_vec_shape**2), t, full_output=True)
            Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (self.__state_vec_shape, self.__state_vec_shape)) for i in range(len(Phi_deltas))]
            if self.intrinsic_noise:
                Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
                integrands = np.array([np.dot(np.dot(a, self.intrinsic_noise), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
                integrand_split = [map(list, zip(*integrands.reshape(*integrands.shape[:1], -1)))]
                # calculate integral numerically using simpsons rule
                self.discrete_intrinsic_noise = np.reshape(np.array([simps(i, t) for i in integrand_split]), (self.__state_vec_shape, self.__state_vec_shape))
            self.discrete_transition_matrix = Phi_s_matrix_form[1]
            return
        else:
            raise RuntimeError("This dynamical model is already discrete!")

    def time_inv_compute_discrete_transition_matrix(self, from_time, to_time):
        """Returns solution for Phi from t_k to t_k+dt_filter for an time invariant transition matrix.
         Can be also used for very slowly varying functions.
        :param to_time: float
        :param from_time: start time of the current step (t_k)
        :return: exp(F*dt)
        """
        if self.__is_model_differential and self.__discretization_active and self.__is_model_time_invariant:
            self.discrete_transition_matrix = expm(self.evaluate_transition_matrix_at_time_t(from_time) * (to_time - from_time))
        return self.discrete_transition_matrix

    def step(self, state_mean, state, time):
        raise NotImplementedError('step function not implemented')


class LinearDifferentialDynamicalModel(DynamicalModel):

    def __init__(self, transition_matrix, dt, intrinsic_noise=None, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        DynamicalModel.__init__(self, transition_matrix, intrinsic_noise=intrinsic_noise, dt=dt)

    def step(self, state_mean, state, time):
        self._logger.debug('Performing a step for time %r' % str(time))
        state_mean.update(state_mean.vec + self.evaluate_transition_matrix_at_time_t(time).dot(
            state_mean.vec) * self.dt)
        if self.intrinsic_noise:
            state.update(state_mean.vec + self.intrinsic_noise.step())
        else:
            state.update(state_mean)
        return
