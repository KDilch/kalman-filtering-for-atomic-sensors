from types import SimpleNamespace
import numpy as np

config = SimpleNamespace()

config.physical_parameters = {
    'larmour_freq': 6.,
    'spin_correlation_const': 0.33,
    'light_correlation_const': 1.
}

config.coupling = {
    'omega_p': [6., 0.],
    'g_p': 150.,
    'phase_shift': 0.
}

config.simulation = {
    'number_periods': 10.,
    'dt_sensor': 0.005,
    'spin_y_initial_val': 2.,
    'spin_z_initial_val': 2.,
    'q_initial_val': 2.,
    'p_initial_val': 2.
}

config.filter = {
    'dt_filter': 0.01,
    'spin_y_initial_val': None,
    'spin_z_initial_val': None,
    'q_initial_val': None,
    'p_initial_val': None,
    'P0': None
}

config.noise_and_measurement = {
    'QJy': 0.01,
    'QJz': 0.01,
    'Qq': 0.01,
    'Qp': 0.01,
    'gD': 100.,
    'QD': 0.01
}

config.W = {
    'W_jy': np.array([[1., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.]]),
    'W_jz': np.array([[0., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.]]),
    'W_q': np.array([[0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 0.]]),
    'W_p': np.array([[0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 1.]])
}
