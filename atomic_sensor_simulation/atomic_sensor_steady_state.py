#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sympy import *
from scipy.linalg import solve_discrete_are, expm

from atomic_sensor_simulation.utilities import eval_matrix_of_functions
from config import config

R = np.array([[lambda t: 1., lambda t: 0., lambda t: 0., lambda t: 0.],
              [lambda t: 0., lambda t: 1., lambda t: 0., lambda t: 0.],
              [lambda t: 0., lambda t: 0., lambda t: np.cos(config.coupling['omega_p'] * t + config.coupling['phase_shift']),
                lambda t: np.sin(config.coupling['omega_p'] * t + config.coupling['phase_shift'])],
              [lambda t: 0., lambda t: 0., lambda t: -np.sin(config.coupling['omega_p'] * t + config.coupling['phase_shift']),
               lambda t: np.cos(config.coupling['omega_p'] * t + config.coupling['phase_shift'])]])
steady_cov_predict_RF = None
steady_cov_update_RF = None
R_T = R.transpose()

def compute_steady_state_solution_for_atomic_sensor(t, F, model):
    global steady_cov_predict_RF
    global steady_cov_update_RF
    if steady_cov_predict_RF is None or steady_cov_update_RF is None:
        F_RF = change_reference_frame_rotating(F,
                                               eval_matrix_of_functions(R, t),
                                               eval_matrix_of_functions(R_T, t))
        R_delta = model.R_delta
        Phi_delta_RF = expm(F_RF * model.dt)
        Q_delta = compute_Q_delta_sympy(F_RF, Matrix(model.Q), model.dt, 50)
        steady_cov_predict_RF = solve_discrete_are(a=np.transpose(Phi_delta_RF),
                                                b=np.transpose(model.H),
                                                r=R_delta,
                                                q=Q_delta)
        S_steady = model.R_delta + np.dot(np.dot(model.H, steady_cov_predict_RF), np.transpose(model.H))
        K_steady = np.dot(np.dot(steady_cov_predict_RF, np.transpose(model.H)), np.linalg.inv(S_steady))
        steady_cov_update_RF = np.dot((np.identity(model.dim_x) - np.dot(K_steady, model.H)), steady_cov_predict_RF)

    # go back to LAB reference frame
    steady_cov_predict = change_reference_frame_rotating(steady_cov_predict_RF,
                                                         R=eval_matrix_of_functions(R_T, t),
                                                         R_T=eval_matrix_of_functions(R, t))
    steady_cov_update = change_reference_frame_rotating(steady_cov_update_RF,
                                                        R=eval_matrix_of_functions(R_T, t),
                                                        R_T=eval_matrix_of_functions(R, t))

    return steady_cov_predict, steady_cov_update

def change_reference_frame_rotating(object, R, R_T):
    return np.dot(np.dot(R, object), R_T)

def compute_Q_delta_sympy(F_RF, Q, delta_t, num_terms):
    Phi_delta_t = expm(F_RF*delta_t)
    integral = np.dot(np.dot(Phi_delta_t, Q), Phi_delta_t.T)*delta_t
    return np.array(integral).astype(np.float64)

def compute_expm_approx(matrix, num_terms):
    out = zeros(*(matrix.shape))
    for n in range(num_terms):
        out += matrix ** n / factorial(n)
    return out

