#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sympy import *
from numpy.linalg import matrix_power
from control import dare
from scipy.linalg import solve_discrete_are, expm
from scipy.integrate import simps
from scipy.integrate import odeint


from atomic_sensor_simulation.utilities import eval_matrix_of_functions, differentiate_matrix_of_functions
from config import config

R = np.array([[lambda t: 1., lambda t: 0., lambda t: 0., lambda t: 0., lambda t: 0],
              [lambda t: 0., lambda t: 1., lambda t: 0., lambda t: 0., lambda t: 0],
              [lambda t: 0., lambda t: 0., lambda t: np.cos(config.coupling['omega_p'] * t + config.coupling['phase_shift']),
                lambda t: np.sin(config.coupling['omega_p'] * t + config.coupling['phase_shift']), lambda t:0],
              [lambda t: 0., lambda t: 0., lambda t: -np.sin(config.coupling['omega_p'] * t + config.coupling['phase_shift']),
               lambda t: np.cos(config.coupling['omega_p'] * t + config.coupling['phase_shift']), lambda t:0],
              [lambda t:0, lambda t:0, lambda t:0, lambda t:0, lambda t:0]])
# steady_cov_predict_RF = None
# steady_cov_update_RF = None
R_T = R.transpose()

def compute_steady_state_solution_for_atomic_sensor(t, F, model):
    # global steady_cov_predict_RF
    # global steady_cov_update_RF
    R_derivative = differentiate_matrix_of_functions(R, t)
    # if steady_cov_predict_RF is None or steady_cov_update_RF is None:
    F_RF = change_reference_frame_rotating(F,
                                           eval_matrix_of_functions(R, t),
                                           eval_matrix_of_functions(R_T, t),
                                           R_derivative)
    R_delta = model.R_delta
    Phi_delta_RF = expm(F_RF * model.dt)
    Phi_RF = expm(F_RF*t)
    Q_delta = compute_Q_delta_sympy(F_RF, Matrix(model.Q), model.dt)

    steady_cov_predict_RF = solve_discrete_are(a=np.transpose(Phi_delta_RF),
                                            b=np.transpose(model.H),
                                            r=R_delta,
                                            q=Q_delta)
    # print('steady cov prediction', steady_cov_predict_RF)
    S_steady = model.R_delta + np.dot(np.dot(model.H, steady_cov_predict_RF), np.transpose(model.H))
    K_steady = np.dot(np.dot(steady_cov_predict_RF, np.transpose(model.H)), np.linalg.inv(S_steady))
    steady_cov_update_RF = np.dot((np.identity(model.dim_x) - np.dot(K_steady, model.H)), steady_cov_predict_RF)

    # go back to LAB reference frame
    steady_cov_predict = change_reference_frame_rotating1(steady_cov_predict_RF,
                                                         R=eval_matrix_of_functions(R_T, t),
                                                         R_T=eval_matrix_of_functions(R, t),
                                                         R_derrrivative=R_derivative)
    steady_cov_update = change_reference_frame_rotating1(steady_cov_update_RF,
                                                        R=eval_matrix_of_functions(R_T, t),
                                                        R_T=eval_matrix_of_functions(R, t),
                                                         R_derrrivative =R_derivative)

    return steady_cov_predict, steady_cov_update

def change_reference_frame_rotating(object, R, R_T, R_derrrivative):
    return np.dot(np.dot(R, object), R_T)+np.dot(R_derrrivative, R_T)

def change_reference_frame_rotating1(object, R, R_T, R_derrrivative):
    return np.dot(np.dot(R, object), R_T)

def compute_Q_delta_sympy(F_RF, Q, delta_t, num_terms=30):
    # #Approx exp with Taylor expansion //not so great
    # out = zeros(*(F_RF.shape))
    # for n in range(num_terms):
    #     matrix_to_n = matrix_power(F_RF, n) / factorial(n)
    #     Phi_Q_Phi_t_Nth_term = np.dot(np.dot(matrix_to_n, Q), np.transpose(matrix_to_n))
    #     matrix_flat = Phi_Q_Phi_t_Nth_term.flatten()
    #     shape = np.shape(Phi_Q_Phi_t_Nth_term)
    #     matrix = np.empty_like(matrix_flat)
    #     for index, element in np.ndenumerate(matrix_flat):
    #         matrix[index] = lambda t: t ** (2 * n) * element
    #     Phi_Q_PHI_T = np.reshape(matrix, shape)
    #     from utilities import integrate_matrix_of_functions
    #     int = integrate_matrix_of_functions(Phi_Q_PHI_T, from_x=0, to_x=delta_t)
    #     Phi_delta_RF = expm(F_RF * delta_t)
    #     out += np.dot(np.dot(Phi_delta_RF, int), np.transpose(Phi_delta_RF))
    # return np.array(out).astype(np.float64)
    #Numerical integrals
    def Phi_t(t):
        return expm(F_RF*t)
    t = np.linspace(0, delta_t, num=num_terms)
    Phi_deltas = np.array([Phi_t(i) for i in t])
    Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (5, 5)) for i in range(len(Phi_deltas))]
    Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
    integrands = np.array([np.dot(np.dot(a, Q), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
    # Assuming 4x4 matrices! #TODO get rid of it
    a = integrands.reshape(-1, integrands.shape[-1])
    int00, int01, int02, int03, int04, int10, int11, int12, int13, int14, int20, int21, int22, int23, int24, int30, int31, int32, int33, int34, int40, int41, int42, int43, int44 = map(
        list, zip(*integrands.reshape(*integrands.shape[:1], -1)))
    integrand_split = [int00, int01, int02, int03, int04, int10, int11, int12, int13, int14, int20, int21, int22, int23, int24, int30, int31,
                       int32, int33, int34, int40, int41, int42, int43, int44]
    Q_delta = np.dot(np.dot(Phi_t(delta_t), np.reshape(np.array([simps(i, t) for i in integrand_split]), (5, 5))), np.transpose(Phi_t(delta_t)))
    return np.array(Q_delta, dtype=np.float64)

def compute_expm_approx(matrix, num_terms):
    out = zeros(*(matrix.shape))
    for n in range(num_terms):
        out += matrix ** n / factorial(n)
    return out

def create_matrix_of_fuctions_Phi_Q_Phi_T(matrix, Q, dt, num_terms):
    out = zeros(*(matrix.shape))
    for n in range(num_terms):
        matrix_to_n = matrix ** n / factorial(n)
        Phi_Q_Phi_t_Nth_term = np.dot(np.dot(matrix_to_n, Q), np.transpose(matrix_to_n))
        matrix_flat = Phi_Q_Phi_t_Nth_term.flatten()
        shape = np.shape(Phi_Q_Phi_t_Nth_term)
        matrix = np.empty_like(matrix_flat)
        for index, element in np.ndenumerate(matrix_flat):
            matrix[index] = lambda t: t ** (2*n) * element
        Phi_Q_PHI_T = np.reshape(matrix, shape)
        from utilities import integrate_matrix_of_functions
        int=integrate_matrix_of_functions(Phi_Q_PHI_T, from_x=0, to_x=dt)
        Phi_delta_RF = expm(matrix * dt)
        out += np.dot(np.dot(Phi_delta_RF, int),np.transpose(Phi_delta_RF))
    return out

