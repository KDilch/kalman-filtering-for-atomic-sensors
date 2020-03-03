from atomic_sensor_simulation.utilities import operable
from scipy.signal import square
import numpy as np


def create_operable_const_func(const):
    @operable
    def const_func(t):
        return const
    return const_func


def create_operable_sin_func(amplitude, omega, phase_shift):
    @operable
    def sin_func(t):
        return amplitude*np.sin(omega*t+phase_shift)
    return sin_func


def create_operable_cos_func(amplitude, omega, phase_shift):
    @operable
    def cos_func(t):
        return amplitude*np.cos(omega*t+phase_shift)
    return cos_func

def create_operable_step_func(time_arr):
    @operable
    def square(t):
        import copy
        t_copy = copy.deepcopy(t)
        print(t_copy, "t_copy")
        index = np.where(time_arr == t_copy)
        print(index[0], "index")
        return square(time_arr)[index[0]]
    return square
