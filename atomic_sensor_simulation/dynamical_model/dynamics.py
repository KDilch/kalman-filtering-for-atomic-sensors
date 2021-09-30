# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC
import logging
from scipy.linalg import expm
from scipy.integrate import odeint, simps, trapz, quad


class DynamicalModel(ABC):
    """Class holding a matrix of functions - linear/ non-linear, discrete/continuous, time-dependent/time independent"""

    def __init__(self,
                 transition_matrix,
                 dt,
                 intrinsic_noise=None,
                 is_model_differential=True,
                 discretization_active=False,
                 is_model_time_invariant=False,
                 discrete_dt=None,
                 initial_time=None
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
            if initial_time is None:
                raise ValueError("Initial time not defined - the discretization cannot be performed.")
            else:
                self._discrete_transition_matrix = None
                self._discrete_intrinsic_noise = None
                self._discrete_dt = discrete_dt
                self.num_compute_discrete_transition_and_noise_matrices(initial_time,
                                                                        time_resolution=30)

    def evaluate_transition_matrix_at_time_t(self, time):
        """Evaluates the matrix of functions at time t"""
        matrix_flat = self.transition_matrix.flatten()
        shape = np.shape(self.transition_matrix)
        evaluated_matrix = np.empty_like(matrix_flat)
        for index, element in np.ndenumerate(matrix_flat):
            evaluated_matrix[index] = matrix_flat[index](time)
        return np.reshape(evaluated_matrix, shape)

    def num_compute_discrete_transition_and_noise_matrices(self, from_time, time_resolution=10):
        """Computes discrete model (Phi, Q^{\Delta})"""
        if self.__is_model_differential and self.__discretization_active:
            Phi_0 = np.reshape(np.identity(self.__state_vec_shape),
                               self.__state_vec_shape ** 2)

            def dPhidt(Phi, t):
                return np.reshape(np.dot(np.array(self.evaluate_transition_matrix_at_time_t(time=t), dtype=float),
                                         np.reshape(Phi, (self.__state_vec_shape, self.__state_vec_shape))),
                                  self.__state_vec_shape ** 2)

            t = np.linspace(from_time, from_time + self._discrete_dt, num=time_resolution)  # times to report solution

            for i in range(1, time_resolution):
                # span for next time step
                tspan = [t[i - 1], t[i]]
                # solve for next step
                Phi = odeint(dPhidt, Phi_0, tspan)
                # next initial condition
                Phi_0 = Phi[1]
            self._discrete_transition_matrix = np.reshape(Phi_0, (self.__state_vec_shape, self.__state_vec_shape))


            Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, 16), t, full_output=True)
            # Numerical
            Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (4, 4)) for i in range(len(Phi_deltas))]
            Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
            integrands = np.array(
                [np.dot(np.dot(a, self.intrinsic_noise.cov), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
            int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23, int30, int31, int32, int33 = map(
                list, zip(*integrands.reshape(*integrands.shape[:1], -1)))
            integrand_split = [int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23,
                               int30,
                               int31, int32, int33]
            # calculate integral numerically using simpsons rule
            self._discrete_intrinsic_noise = np.reshape(np.array([simps(i, t) for i in integrand_split]), (4, 4))
            return

            # def dPhidt(Phi, t):
            #     return np.reshape(np.dot(np.array(self.evaluate_transition_matrix_at_time_t(time=t), dtype=float),
            #                              np.reshape(Phi, (self.__state_vec_shape, self.__state_vec_shape))),
            #                       self.__state_vec_shape ** 2)
            #
            # t = np.linspace(from_time, from_time+self._discrete_dt, num=time_resolution)  # times to report solution
            # solve ODE
            # Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, self.__state_vec_shape**2), t, full_output=True)
            # Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (self.__state_vec_shape, self.__state_vec_shape)) for i in range(len(Phi_deltas))]
            # # self._discrete_transition_matrix = Phi_s_matrix_form[-1]
            # if self.intrinsic_noise:
            #     Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
            #     Q_delta_integrands = np.array([np.dot(np.dot(Phi, self.intrinsic_noise.cov), PhiT) for Phi, PhiT in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
            #     Q_delta_integrand_split = list(map(list, zip(*Q_delta_integrands.reshape(*Q_delta_integrands.shape[:1], -1))))
            #     self._discrete_intrinsic_noise = np.reshape(np.array([trapz(i, t) for i in Q_delta_integrand_split]), (self.__state_vec_shape, self.__state_vec_shape))
            # return
        else:
            raise RuntimeError("Discretization unsuccessful.")

    @property
    def discrete_transition_matrix(self):
        return self._discrete_transition_matrix

    @property
    def discrete_transition_matrix_T(self):
        return np.transpose(self._discrete_transition_matrix)

    @property
    def discrete_intrinsic_noise(self):
        return self._discrete_transition_matrix

    @property
    def discrete_dt(self):
        return self._discrete_dt

    @property
    def state_vec_shape(self):
        return self.__state_vec_shape

    def time_inv_compute_discrete_transition_matrix(self, from_time, to_time):
        """Returns solution for Phi from t_k to t_k+dt_filter for an time invariant transition matrix.
         Can be also used for very slowly varying functions.
        :param to_time: float
        :param from_time: start time of the current step (t_k)
        :return: exp(F*dt)
        """
        if self.__is_model_differential and self.__discretization_active and self.__is_model_time_invariant:
            self._discrete_transition_matrix = expm(self.evaluate_transition_matrix_at_time_t(from_time) * (to_time - from_time))
        return self.discrete_transition_matrix

    def step(self, state_mean, state, time):
        raise NotImplementedError('step function not implemented')


class LinearDifferentialDynamicalModel(DynamicalModel):

    def __init__(self,
                 transition_matrix,
                 dt,
                 intrinsic_noise=None,
                 logger=None,
                 discretization_active=False,
                 is_model_time_invariant=False,
                 initial_time=None,
                 discrete_dt=None):
        self._logger = logger or logging.getLogger(__name__)
        DynamicalModel.__init__(self,
                                transition_matrix=transition_matrix,
                                intrinsic_noise=intrinsic_noise,
                                dt=dt,
                                is_model_differential=True,
                                discretization_active=discretization_active,
                                is_model_time_invariant=is_model_time_invariant,
                                initial_time=initial_time,
                                discrete_dt=discrete_dt)

    def step(self, state_mean, state, time):
        self._logger.debug('Performing a step for time %r' % str(time))
        state_mean.update(state_mean.vec + (self.evaluate_transition_matrix_at_time_t(time).dot(
            state_mean.vec)) * self.dt)
        if self.intrinsic_noise:
            state.update(state_mean.vec + self.intrinsic_noise.step())
        else:
            state.update(state_mean)
        return
