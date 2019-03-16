# -*- coding: utf-8 -*-
import filterpy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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