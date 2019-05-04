# -*- coding: utf-8 -*-
import filterpy
import numpy as np
from scipy import integrate
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from atomic_sensor_simulation import CONSTANTS
from atomic_sensor_simulation.operable_functions import create_multiplicable_const_func


def initialize_kalman_filter_from_derrivatives(initial_state_vec):
    dt = 1.
    atoms_correlation_const = 0.000001
    # d_transition_matrix = np.array([[-atoms_correlation_const, CONSTANTS.g_a_COUPLING_CONST], [0, 1.]])
    F = np.exp(np.array([[-atoms_correlation_const*dt, dt * CONSTANTS.g_a_COUPLING_CONST], [0, dt]]))

    d_transition_matrix = np.array([[create_multiplicable_const_func(-atoms_correlation_const), create_multiplicable_const_func(CONSTANTS.g_a_COUPLING_CONST)], [create_multiplicable_const_func(0), create_multiplicable_const_func(1.)]])
    kfilter = KalmanFilter(dim_x=2, dim_z=1)
    kfilter.x = initial_state_vec
    kfilter.F = np.exp(np.array([[-atoms_correlation_const*dt, dt * CONSTANTS.g_a_COUPLING_CONST], [0, dt]]))
    kfilter.H = np.array([[CONSTANTS.g_d_COUPLING_CONST, 0]])
    kfilter.P *= CONSTANTS.SCALAR_STREGTH_y
    kfilter.R = np.array([[CONSTANTS.SCALAR_STREGTH_y/dt]])
    # kfilter.B = compute_B_from_d_vals(d_transition_matrix, np.eye(2), np.zeros(2).T, 1, 1)
    kfilter.B = F.dot(np.eye(2))
    from filterpy.common import Q_discrete_white_noise
    kfilter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    dQ = np.array([[create_multiplicable_const_func(CONSTANTS.SCALAR_STREGTH_j), create_multiplicable_const_func(0)], [create_multiplicable_const_func(0), create_multiplicable_const_func(CONSTANTS.SCALAR_STRENGTH_q)]])
    # kfilter.Q = compute_Q_from_d_vals(d_transition_matrix, np.array([[create_multiplicable_const_func(1), create_multiplicable_const_func(0)], [create_multiplicable_const_func(0), create_multiplicable_const_func(1)]]), dQ, dt, dt)
    kfilter.Q = F.dot(np.eye(2)).dot(np.array([[(CONSTANTS.SCALAR_STREGTH_j), (0)], [(0), (CONSTANTS.SCALAR_STRENGTH_q)]])).dot(np.eye(2).T).dot(F.T)
    return kfilter


def compute_B_from_d_vals(d_transition_matrix, d_control_transition_matrix, time):
    #assuming there F, Gamma aren't time dependent
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
    filter = KalmanFilter(dim_x=2, dim_z=1)
    #assign initial values
    filter.x = initial_x
    filter.F = transition_matrix
    filter.H = measurement_function
    filter.P *= uncertainty #cov matrix
    filter.R = 5
    filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    return filter

def step():
    pass

def predict():
    pass