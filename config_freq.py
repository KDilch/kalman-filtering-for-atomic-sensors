from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.physical_parameters = {
        'larmour_freq': 6.,
        'spin_correlation_const': 0.33,
        'light_correlation_const': 1.
    }

config.coupling = {
    'omega_p': 15.,
    'g_p': 50.,
    'phase_shift': 0.
}

config.simulation = {
    'number_periods': 6.,
    'dt_sensor': 0.005,
    'x1': 2.,
    'x2': 2.,
    'x3': 0.1
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
    'Qx1': 0.,
    'Qx2': 0.,
    'Qx3': 0.0001,
    'R': 0.0001
}

