# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from atomic_sensor_simulation import CONSTANTS
from atomic_sensor_simulation.utilities import exp_matrix_of_functions, create_matrix_of_functions
from atomic_sensor_simulation.operable_functions import create_operable_cos_func, create_operable_const_func, create_operable_sin_func


def initialize_kalman_filter_from_derrivatives(initial_state_vec, dt, initial_F):
    F = np.array(initial_F, dtype='float64')
    kfilter = KalmanFilter(dim_x=4, dim_z=2)
    kfilter.x = initial_state_vec
    kfilter.F = F
    kfilter.H = np.array([[1./0.3048, 0., 0., 0.], [0., 0., 1./0.3048, 0.]], dtype='float64')
    kfilter.P = np.eye(4) * 500.
    kfilter.R = np.eye(2) * 5.
    # kfilter.B = F.dot(np.eye(2))
    # kfilter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=np.sqrt(CONSTANTS.SCALAR_STREGTH_j))
    # kfilter.Q = F.dot(np.eye(2)).dot(np.array([[(CONSTANTS.SCALAR_STREGTH_j), (0)], [(0), (CONSTANTS.SCALAR_STRENGTH_q)]])).dot(np.eye(2).T).dot(F.T)
    # kfilter.Q = np.array([[(CONSTANTS.SCALAR_STREGTH_j), (0)], [(0), (CONSTANTS.SCALAR_STRENGTH_q)]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
    kfilter.Q = block_diag(q, q)
    return kfilter


def compute_B_from_d_vals(d_transition_matrix, d_control_transition_matrix, time):
    b = d_transition_matrix.dot(d_control_transition_matrix)
    b_flat = b.flatten()
    results_flat = np.empty_like(b_flat)
    for element in range(b_flat.shape[0]):
        results_flat[element] = b_flat[element](time)
        # results_flat[element] = integrate.quad(b_flat[element], time-delta_t, time)[0]
    return results_flat.reshape(2, 2).astype(np.float64)


def compute_Q_from_d_vals(d_transition_matrix, d_noise_transition_matrix, d_Q_matrix, time, delta_t):
    a = d_transition_matrix.dot(d_noise_transition_matrix).dot(d_Q_matrix).dot(d_noise_transition_matrix.T).dot(d_transition_matrix.T)
    a_flat = a.flatten()
    results_flat = np.empty_like(a_flat)
    for element in range(a_flat.shape[0]):
        # results_flat[element] = integrate.quad(a_flat[element], time-delta_t, time)[0]
        results_flat[element] = a_flat[element](time)*(delta_t)
    return results_flat.reshape(2, 2).astype(np.float64)


def initialize_kalman_filter(initial_x, uncertainty, transition_matrix, measurement_function):
    kfilter = KalmanFilter(dim_x=2, dim_z=1)
    kfilter.x = initial_x
    kfilter.F = transition_matrix
    kfilter.H = measurement_function
    kfilter.P *= uncertainty  # covariance matrix
    kfilter.R = 5
    kfilter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    return kfilter
