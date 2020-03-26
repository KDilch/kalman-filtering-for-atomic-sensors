from types import SimpleNamespace
config = SimpleNamespace()

config.simulation = {
    'num_iter_sensor': 200,
    'number_periods': 30,
    'dt_sensor': 1.,
    'x1': 1.,
    'x2': 0.,
    'x3': 0.471239
}

config.filter = {
    'dt_filter': 1.
}

config.noise_and_measurement = {
    'Qx1': 0.,
    'Qx2': 0.,
    'Qx3': 0.01,
    'R': 0.01
}

