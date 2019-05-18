from atomic_sensor_simulation.utilities import operable
import numpy as np


def create_operable_const_func(const):
    @operable
    def const_func(t):
        return const
    return const_func


def create_operable_sin_func(amplitude, omega):
    @operable
    def sin_func(t):
        return amplitude*np.sin(omega*t)
    return sin_func


def create_operable_cos_func(amplitude, omega):
    @operable
    def cos_func(t):
        return amplitude*np.cos(omega*t)
    return cos_func
