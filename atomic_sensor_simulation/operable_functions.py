from atomic_sensor_simulation.utilities import operable
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

def create_operable_sin_func_freq_fluctuations(amplitude, omega_func, phase_shift):
    @operable
    def sin_func(t):
        return amplitude*np.sin(omega_func(t)*t+phase_shift)
    return sin_func


def create_operable_cos_func_freq_fluctuations(amplitude, omega_func, phase_shift):
    @operable
    def cos_func(t):
        return amplitude*np.cos(omega_func(t)*t+phase_shift)
    return cos_func
