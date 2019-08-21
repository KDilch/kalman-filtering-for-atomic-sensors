from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.physical_parameters = {
        'larmour_freq': 6.,
        'spin_correlation_const': 0.33,
        'light_correlation_const': 0.2
    }

config.coupling = {
    'omega_p': 6.0,
    'g_p': 30.,
    'phase_shift': 0.0
}

config.simulation = {
    'number_periods': 10.,
    'dt_sensor': 0.01,
    'spin_y_initial_val': 2.,
    'spin_z_initial_val': 2.,
    'q_initial_val': 0.,
    'p_initial_val': 0.
}

config.filter = {
    'dt_filter': 0.02,
    'spin_y_initial_val': None,
    'spin_z_initial_val': None,
    'q_initial_val': None,
    'p_initial_val': None,
    'P0': None
}

config.noise_and_measurement = {
    'QJy': 0.1,
    'QJz': 0.1,
    'Qq': 0.05,
    'Qp': 0.02,
    'gD': 100,
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