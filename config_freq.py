from types import SimpleNamespace
config = SimpleNamespace()

config.simulation = {
    'num_iter_sensor': 300,
    'number_periods': 30,
    'dt_sensor': 0.05,
    'x1': 1.,
    'x2': 1.,
    'x3': 0.
}

config.filter = {
    'dt_filter': 0.05
}

config.noise_and_measurement = {
    'Qx1': 0.,
    'Qx2': 0.,
    'Qx3': 0.001,
    'R': 0.0001
}

