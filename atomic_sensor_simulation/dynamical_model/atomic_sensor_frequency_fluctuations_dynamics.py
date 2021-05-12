# -*- coding: utf-8 -*-
from atomic_sensor_simulation.dynamical_model.dynamics import LinearDifferentialDynamicalModel
from atomic_sensor_simulation.operable_functions import create_operable_const_func, create_operable_cos_func_freq_fluctuations, create_operable_sin_func_freq_fluctuations
import logging
import numpy as np


class AtomicSensorLinearDifferentialDynamicalModelFrequencyFluctuations(LinearDifferentialDynamicalModel):

    def __init__(self,
                 larmour_freq_func,
                 logger=None,
                 **kwargs
                 ):
        self._logger = logger or logging.getLogger(__name__)
        light_correlation_const = kwargs['light_correlation_const']
        coupling_amplitude = kwargs['coupling_amplitude']
        coupling_phase_shift = kwargs['coupling_phase_shift']
        larmour_freq = kwargs['larmour_freq']
        spin_correlation_const = kwargs['spin_correlation_const']
        transition_matrix = np.array([[create_operable_const_func(-spin_correlation_const),
                                       create_operable_const_func(larmour_freq),
                                       create_operable_const_func(0),
                                       create_operable_const_func(0)],

                                      [create_operable_const_func(-larmour_freq),
                                       create_operable_const_func(-spin_correlation_const),
                                       create_operable_cos_func_freq_fluctuations(amplitude=coupling_amplitude,
                                                                                  omega_func=larmour_freq_func,
                                                                                  phase_shift=coupling_phase_shift),
                                       create_operable_sin_func_freq_fluctuations(amplitude=coupling_amplitude,
                                                                                  omega_func=larmour_freq_func,
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


class AtomicSensorLinearDifferentialDynamicalModelFrequencyFluctuationsExpandedDims(LinearDifferentialDynamicalModel):
    #TODO implement this
    def __init__(self,
                 larmour_freq_func,
                 logger=None,
                 **kwargs
                 ):
        self._logger = logger or logging.getLogger(__name__)
        light_correlation_const = kwargs['light_correlation_const']
        coupling_amplitude = kwargs['coupling_amplitude']
        coupling_phase_shift = kwargs['coupling_phase_shift']
        larmour_freq = kwargs['larmour_freq']
        spin_correlation_const = kwargs['spin_correlation_const']
        transition_matrix = np.array([[create_operable_const_func(-spin_correlation_const),
                                       create_operable_const_func(larmour_freq),
                                       create_operable_const_func(0),
                                       create_operable_const_func(0)],

                                      [create_operable_const_func(-larmour_freq),
                                       create_operable_const_func(-spin_correlation_const),
                                       create_operable_cos_func_freq_fluctuations(amplitude=coupling_amplitude,
                                                                                  omega_func=larmour_freq_func,
                                                                                  phase_shift=coupling_phase_shift),
                                       create_operable_sin_func_freq_fluctuations(amplitude=coupling_amplitude,
                                                                                  omega_func=larmour_freq_func,
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
