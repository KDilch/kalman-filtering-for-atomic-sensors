#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import solve_discrete_are, expm
from scipy.integrate import odeint, simps, trapz, quad

from atomic_sensor_simulation.utilities import eval_matrix_of_functions, differentiate_matrix_of_functions


class SteadyStateSolver(object):

    def __init__(self,
                 kalmanfilter):
        self._kalmanfilter = kalmanfilter

    #TODO Implement a general solver


class AtomicSensorSteadyStateSolver(SteadyStateSolver):

    def __init__(self, kalmanfilter, omega_p, phase_shift):
        SteadyStateSolver.__init__(self, kalmanfilter)
        self.__rotating_frame_transform = np.array([[lambda t: 1., lambda t: 0., lambda t: 0., lambda t: 0.],
                                                    [lambda t: 0., lambda t: 1., lambda t: 0., lambda t: 0.],
                                                    [lambda t: 0., lambda t: 0., lambda t: np.cos(omega_p * t + phase_shift), lambda t: np.sin(omega_p * t + phase_shift)],
                                                    [lambda t: 0., lambda t: 0., lambda t: -np.sin(omega_p * t + phase_shift), lambda t: np.cos(omega_p * t + phase_shift)]])
        self.__rotating_frame_transform_T = self.__rotating_frame_transform.transpose()
        self.__D_rotating_frame_transform = None
        self.__F_RF = None
        self.__Phi_delta_RF = None
        self.__steady_prior = None
        self.__steady_post = None
        self.__Q_delta_RF = None

    def steady_state_solution_rotating_frame(self, t):
        self.__D_rotating_frame_transform = differentiate_matrix_of_functions(self.__rotating_frame_transform, t)
        self.__F_RF = self.__change_time_dep_reference_frame_to_rotating(self._kalmanfilter.continuous_transition_matrix,
                                                                           eval_matrix_of_functions(self.__rotating_frame_transform, t),
                                                                           eval_matrix_of_functions(self.__rotating_frame_transform_T, t),
                                                                           self.__D_rotating_frame_transform)
        if self.__Q_delta_RF is None:
            self.__num_compute_Q_delta_in_RF(t)
        if self.__Phi_delta_RF is None:
            self.__Phi_delta_RF = expm(self.__F_RF * self._kalmanfilter.dt)
        steady_cov_predict_RF = solve_discrete_are(a=np.transpose(self.__Phi_delta_RF),
                                                   b=np.transpose(self._kalmanfilter.measurement_matrix),
                                                   r=self._kalmanfilter.discrete_measurement_noise_matrix,
                                                   q=self.__Q_delta_RF)
        S_steady = self._kalmanfilter.discrete_measurement_noise_matrix + np.dot(np.dot(self._kalmanfilter.measurement_matrix, steady_cov_predict_RF), np.transpose(self._kalmanfilter.measurement_matrix))
        K_steady = np.dot(np.dot(steady_cov_predict_RF, np.transpose(self._kalmanfilter.measurement_matrix)), np.linalg.inv(S_steady))
        steady_cov_update_RF = np.dot((np.identity(self._kalmanfilter.state_vec_shape) - np.dot(K_steady, self._kalmanfilter.measurement_matrix)), steady_cov_predict_RF)

        # go back to NOT ROTATING reference frame
        self.__steady_prior = self.__change_time_indep_reference_frame_rotating(steady_cov_predict_RF,
                                                                                R=eval_matrix_of_functions(self.__rotating_frame_transform_T, t),
                                                                                R_T=eval_matrix_of_functions(self.__rotating_frame_transform, t))
        self.__steady_post = self.__change_time_indep_reference_frame_rotating(steady_cov_update_RF,
                                                                               R=eval_matrix_of_functions(self.__rotating_frame_transform_T, t),
                                                                               R_T=eval_matrix_of_functions(self.__rotating_frame_transform, t))
        return

    def __num_compute_Q_delta_in_RF(self, t, num_terms=30):
        """
        This has to be computed in RF because if just transformed the matrix Q^Delta is not Hermitian -> dare can not
        be solved (this is due to the numerical error).
        :param num_terms:
        :return:
        """
        def Phi_t(t):
            return expm(self.__F_RF * t)

        t = np.linspace(t, t+self._kalmanfilter.dt, num=num_terms)  # since everything is time independent I can perform this calculation once
        Phi_deltas = np.array([Phi_t(i) for i in t])
        Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (self._kalmanfilter.state_vec_shape, self._kalmanfilter.state_vec_shape)) for i in range(len(Phi_deltas))]
        Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
        Q_delta_integrands = np.array([np.dot(np.dot(Phi, self._kalmanfilter.intrinsic_noise_matrix), PhiT) for Phi, PhiT in
                                       zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
        Q_delta_integrand_split = list(map(list, zip(*Q_delta_integrands.reshape(*Q_delta_integrands.shape[:1], -1))))
        self.__Q_delta_RF = np.reshape(np.array([trapz(i, t) for i in Q_delta_integrand_split]),
                                                (self._kalmanfilter.state_vec_shape, self._kalmanfilter.state_vec_shape))
        return

    @staticmethod
    def __change_time_dep_reference_frame_to_rotating(obj, R, R_T, R_derrrivative):
        return np.dot(np.dot(R, obj), R_T) + np.dot(R_derrrivative, R_T)

    @staticmethod
    def __change_time_indep_reference_frame_rotating(obj, R, R_T):
        return np.dot(np.dot(R, obj), R_T)

    @property
    def steady_prior(self):
        return self.__steady_prior

    @property
    def steady_post(self):
        return self.__steady_post
