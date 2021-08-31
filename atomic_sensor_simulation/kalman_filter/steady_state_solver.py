#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import solve_discrete_are, expm
from scipy.integrate import odeint, simps, trapz, quad

from atomic_sensor_simulation.utilities import eval_matrix_of_functions, differentiate_matrix_of_functions
from config import config


class SteadyStateSolver(object):

    def __init__(self,
                 kalmanfilter):
        self._kalmanfilter = kalmanfilter

    #TODO Implement a general solver
R = np.array([[lambda t: 1., lambda t: 0., lambda t: 0., lambda t: 0.],
              [lambda t: 0., lambda t: 1., lambda t: 0., lambda t: 0.],
              [lambda t: 0., lambda t: 0., lambda t: np.cos(config.coupling["omega_p"] * t + config.coupling["phase_shift"]), lambda t: np.sin(config.coupling["omega_p"] * t + config.coupling["phase_shift"])],
              [lambda t: 0., lambda t: 0., lambda t: -np.sin(config.coupling["omega_p"] * t + config.coupling["phase_shift"]), lambda t: np.cos(config.coupling["omega_p"] * t + config.coupling["phase_shift"])]])

R_T = R.transpose()

class AtomicSensorSteadyStateSolver(SteadyStateSolver):

    def __init__(self, kalmanfilter):
        SteadyStateSolver.__init__(self, kalmanfilter)
        self.__R_derrivative = None
        self.__F_RF = None
        self.__Phi_delta_RF = None
        self.__steady_prior = None
        self.__steady_post = None
        self.__Q_delta_RF = None

    def steady_state_solution_rotating_frame(self, t):
        R_derivative = differentiate_matrix_of_functions(R, t)
        # if steady_cov_predict_RF is None or steady_cov_update_RF is None:
        self.__F_RF = self.__change_time_dep_reference_frame_to_rotating(
            self._kalmanfilter.continuous_transition_matrix,
            eval_matrix_of_functions(R, t),
            eval_matrix_of_functions(R_T, t),
            R_derivative)
        R_delta = self._kalmanfilter.R
        Phi_delta_RF = expm(self.__F_RF * self._kalmanfilter.dt)
        Phi_RF = expm(self.__F_RF * t)
        Q_delta = self.__num_compute_Q_delta_in_RF(t)
        # Q_delta = compute_Q_delta_sympy(F_RF, Matrix(model.Q), model.dt)

        steady_cov_predict_RF = solve_discrete_are(a=np.transpose(Phi_delta_RF),
                                                   b=np.transpose(self._kalmanfilter.measurement_matrix),
                                                   r=R_delta,
                                                   q=Q_delta)
        # print('steady cov prediction', steady_cov_predict_RF)
        S_steady = R_delta + np.dot(np.dot(self._kalmanfilter.measurement_matrix, steady_cov_predict_RF), np.transpose(self._kalmanfilter.measurement_matrix))
        K_steady = np.dot(np.dot(steady_cov_predict_RF, np.transpose(self._kalmanfilter.measurement_matrix)), np.linalg.inv(S_steady))
        steady_cov_update_RF = np.dot((np.identity(self._kalmanfilter.state_vec_shape) - np.dot(K_steady, self._kalmanfilter.measurement_matrix)), steady_cov_predict_RF)

        # go back to LAB reference frame
        self.__steady_prior = self.__change_time_indep_reference_frame_rotating(steady_cov_predict_RF,
                                                              R=eval_matrix_of_functions(R_T, t),
                                                              R_T=eval_matrix_of_functions(R, t))
        self.__steady_post = self.__change_time_indep_reference_frame_rotating(steady_cov_update_RF,
                                                             R=eval_matrix_of_functions(R_T, t),
                                                             R_T=eval_matrix_of_functions(R, t))


    def Phi_RF(self, t):
        return expm(self.__F_RF * t)

    def __num_compute_Q_delta_in_RF(self, t, num_terms=30):
        """
        This has to be computed in RF because if just transformed the matrix Q^Delta is not Hermitian -> dare can not
        be solved (this is due to the numerical error).
        :param num_terms:
        :return:
        """
        t = np.linspace(0, self._kalmanfilter.dt, num=num_terms)  # since everything is time independent I can perform this calculation once
        Phi_deltas = np.array([self.Phi_RF(i) for i in t])
        Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (self._kalmanfilter.state_vec_shape, self._kalmanfilter.state_vec_shape)) for i in range(len(Phi_deltas))]
        Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
        Q_delta_integrands = np.array([np.dot(np.dot(Phi, self._kalmanfilter.intrinsic_noise_matrix), PhiT) for Phi, PhiT in
                                       zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
        Q_delta_integrand_split = list(map(list, zip(*Q_delta_integrands.reshape(*Q_delta_integrands.shape[:1], -1))))
        self.__Q_delta_RF = np.reshape(np.array([trapz(i, t) for i in Q_delta_integrand_split]),
                                                (self._kalmanfilter.state_vec_shape, self._kalmanfilter.state_vec_shape))
        return self.__Q_delta_RF

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
