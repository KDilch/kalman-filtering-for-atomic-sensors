from atomic_sensor_simulation.utilities import multiplicable
import numpy as np


def create_multiplicable_const_func(const):
    @multiplicable
    def const_func(t):
        return const
    return const_func


def create_multiplicable_sin_func(amplitude, omega):
    @multiplicable
    def sin_func(t):
        return amplitude*np.sin(omega*t)
    return sin_func

def create_multiplicable_cos_func(amplitude, omega):
    @multiplicable
    def cos_func(t):
        return amplitude*np.cos(omega*t)
    return cos_func

