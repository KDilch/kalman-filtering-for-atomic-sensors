#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sympy import *
from scipy.linalg import solve_discrete_are, expm

from atomic_sensor_simulation.utilities import eval_matrix_of_functions
from config import config

R = np.array([[lambda t: 1., lambda t: 0., lambda t: 0., lambda t: 0.],
              [lambda t: 0., lambda t: 1., lambda t: 0., lambda t: 0.],
              [lambda t: 0., lambda t: 0., lambda t: np.sin(config.coupling['omega_p'] * t + config.coupling['phase_shift']),
                lambda t: np.cos(config.coupling['omega_p'] * t + config.coupling['phase_shift'])],
              [lambda t: 0., lambda t: 0., lambda t: -np.sin(config.coupling['omega_p'] * t + config.coupling['phase_shift']),
               lambda t: np.cos(config.coupling['omega_p'] * t + config.coupling['phase_shift'])]])
R_T = R.transpose()

def compute_steady_state_solution_for_atomic_sensor(t, F, model):
    F_RF = change_reference_frame_rotating(F, eval_matrix_of_functions(R, t), eval_matrix_of_functions(R_T, t))
    Phi_delta_RF = expm(F_RF * model.dt)
    Q_delta = compute_Q_delta_sympy(F_RF, Matrix(model.Q), model.dt, 5)
    steady_cov_predict_RF = solve_discrete_are(a=np.transpose(Phi_delta_RF),
                                            b=np.transpose(model.H),
                                            r=model.R_delta,
                                            q=Q_delta)
    S_steady = model.R_delta + np.dot(np.dot(model.H, steady_cov_predict_RF), np.transpose(model.H))
    K_steady = np.dot(np.dot(steady_cov_predict_RF, np.transpose(model.H)), np.linalg.inv(S_steady))
    steady_cov_update_RF = np.dot(np.identity(model.dim_x) - np.dot(K_steady, model.H), steady_cov_predict_RF)

    #go back to not rotating RF
    steady_cov_predict = change_reference_frame_rotating(steady_cov_predict_RF,
                                                         eval_matrix_of_functions(R, t),
                                                         eval_matrix_of_functions(R_T, t))
    steady_cov_update = change_reference_frame_rotating(steady_cov_update_RF,
                                                        eval_matrix_of_functions(R, t),
                                                        eval_matrix_of_functions(R_T, t))

    return steady_cov_predict, steady_cov_update

def change_reference_frame_rotating(object, R, R_T):
    return np.dot(np.dot(R, object), R_T)

def compute_Q_delta_sympy(F_RF, Q, delta_t, num_terms):
    tau = symbols('tau', real=True, positive=True)
    Phi_tau_matrix = compute_expm_approx(Matrix(F_RF*(-tau)), num_terms)
    Phi_tau_matrix_T = transpose(compute_expm_approx(Matrix(F_RF*(-tau)), num_terms))
    integral = integrate(Phi_tau_matrix*Q*Phi_tau_matrix_T, (tau, 0, delta_t))
    return np.array(integral).astype(np.float64)


def compute_Q_delta_numerically():
    pass

def compute_expm_approx(matrix, num_terms):
    out = zeros(*(matrix.shape))
    for n in range(num_terms):
        out += matrix ** n / factorial(n)
    return out

