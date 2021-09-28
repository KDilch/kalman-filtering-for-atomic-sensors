from types import SimpleNamespace
import numpy as np

config = SimpleNamespace()

config.physical_parameters = {
    'larmour_freq': 6.,
    'spin_correlation_const': 0.33,
    'light_correlation_const': 1.
}

config.coupling = {
    'omega_p': 6.,
    'g_p': 10.,
    'phase_shift': 0.
}

config.simulation = {
    'number_periods': 2.,
    'dt_simulation': 0.001,
    'spin_y_initial_val': 1.,
    'spin_z_initial_val': 1.,
    'q_initial_val': 2.,
    'p_initial_val': 2.,
    'R': np.array([[0.01]]),
    'simulation_type': ['linear', 'sin', 'square', 'sawtooth']  # must be a list or value [NOT ND.ARRAY], input verification according to ATOMIC_SENSOR_DYNAMICS_TYPES in main.py
}

config.sin_waveform = {
    'frequency': 1./3,
    'amplitude': 10.
}

config.sawtooth_waveform = {
    'frequency': 1./3,
    'amplitude': 10.
}

config.square_waveform = {
    'frequency': 1./3,
    'amplitude': 10.
}

config.filter = {
    'dt_filter': 0.01,
    'spin_y_initial_val': None,
    'spin_z_initial_val': None,
    'q_initial_val': None,
    'p_initial_val': None,
    'P0': None,
    'filter_type': ['lkf', 'ukf']
}

config.noise_and_measurement = {
    # Q00 = QJy, Q11=QJz, Q22=Qq, Q33=Qp]
    'Q': np.array([[0.01, 0., 0., 0.],
                  [0., 0.01, 0., 0.],
                  [0., 0., 0.01, 0.],
                  [0., 0., 0., 0.01]]),
    'gD': 100.}

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
