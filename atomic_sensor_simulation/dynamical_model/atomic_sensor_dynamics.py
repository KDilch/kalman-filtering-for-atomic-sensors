# -*- coding: utf-8 -*-
import logging
import numpy as np

from atomic_sensor_simulation.dynamical_model.dynamics import LinearDifferentialDynamicalModel
from atomic_sensor_simulation.operable_functions import create_operable_const_func, create_operable_cos_func, create_operable_sin_func


class AtomicSensorLinearDifferentialDynamicalModel(LinearDifferentialDynamicalModel):

    def __init__(self,
                 logger=None,
                 **kwargs
                 ):
        self._logger = logger or logging.getLogger(__name__)
        light_correlation_const = kwargs['light_correlation_const']
        coupling_amplitude = kwargs['coupling_amplitude']
        coupling_freq = kwargs['coupling_freq']
        coupling_phase_shift = kwargs['coupling_phase_shift']
        larmour_freq = kwargs['larmour_freq']
        spin_correlation_const = kwargs['spin_correlation_const']
        transition_matrix = np.array([[create_operable_const_func(-spin_correlation_const),
                                       create_operable_const_func(larmour_freq),
                                       create_operable_const_func(0),
                                       create_operable_const_func(0)],

                                      [create_operable_const_func(-larmour_freq),
                                       create_operable_const_func(-spin_correlation_const),
                                       create_operable_cos_func(amplitude=coupling_amplitude,
                                                                omega=coupling_freq,
                                                                phase_shift=coupling_phase_shift),
                                       create_operable_sin_func(amplitude=coupling_amplitude,
                                                                omega=coupling_freq,
                                                                phase_shift=coupling_phase_shift)],

                                      [create_operable_const_func(0),
                                       create_operable_const_func(0),
                                       create_operable_const_func(-light_correlation_const),
                                       create_operable_const_func(0)],

                                      [create_operable_const_func(0),
                                       create_operable_const_func(0),
                                       create_operable_const_func(0),
                                       create_operable_const_func(-light_correlation_const)]])

        LinearDifferentialDynamicalModel.__init__(self, transition_matrix=transition_matrix, logger=logger)


class AtomicSensorSinDynamicalModel(AtomicSensorLinearDifferentialDynamicalModel):
    def __init__(self,
                 logger=None,
                 **kwargs
                 ):
        self._logger = logger if logger else logging.getLogger(__name__)
        self.__sin_wave_frequency = kwargs['sin_wave_frequency']
        self.__sin_wave_amplitude = kwargs['sin_wave_amplitude']
        AtomicSensorLinearDifferentialDynamicalModel.__init__(self, **kwargs)

    def step(self, state_mean, state, time, time_step, intrinsic_noise=None):
        LinearDifferentialDynamicalModel.step(self,
                                              state_mean,
                                              state,
                                              time,
                                              time_step,
                                              intrinsic_noise)
        state_mean.vec[2] = self.__sin_wave_func(time+time_step)

    def __sin_wave_func(self, time):
        return np.sin(2*np.pi*time*self.__sin_wave_frequency)


class AtomicSensorSquareWaveDynamicalModel(AtomicSensorLinearDifferentialDynamicalModel):
    def __init__(self,
                 logger=None,
                 **kwargs
                 ):
        self._logger = logger if logger else logging.getLogger(__name__)
        self._square_wave_frequency = kwargs['square_wave_frequency']
        self._square_wave_amplitude = kwargs['square_wave_amplitude']
        AtomicSensorLinearDifferentialDynamicalModel.__init__(self, **kwargs)

    def step(self, state_mean, state, time, time_step, intrinsic_noise=None):
        LinearDifferentialDynamicalModel.step(self,
                                              state_mean,
                                              state,
                                              time,
                                              time_step,
                                              intrinsic_noise)
        state_mean.vec[2] = self.__square_wave_func(time+time_step)

    def __square_wave_func(self, time):
        return self._square_wave_amplitude*np.sign(np.sin(2*np.pi*self._square_wave_frequency*time))


class AtomicSensorSawtoothDynamicalModel(AtomicSensorLinearDifferentialDynamicalModel):
    def __init__(self,
                 logger=None,
                 **kwargs
                 ):
        self._logger = logger if logger else logging.getLogger(__name__)
        self._sawtooth_wave_frequency = kwargs['sawtooth_wave_frequency']
        self._sawtooth_wave_amplitude = kwargs['sawtooth_wave_amplitude']
        AtomicSensorLinearDifferentialDynamicalModel.__init__(self, **kwargs)

    def step(self, state_mean, state, time, time_step, intrinsic_noise=None):
        LinearDifferentialDynamicalModel.step(self,
                                              state_mean,
                                              state,
                                              time,
                                              time_step,
                                              intrinsic_noise)
        state_mean.vec[2] = self.__sawtooth_func(time)

    def __sawtooth_func(self, time):
        return self._sawtooth_wave_amplitude*2*((time*self._sawtooth_wave_frequency)-np.floor(0.5+time*self._sawtooth_wave_frequency))
