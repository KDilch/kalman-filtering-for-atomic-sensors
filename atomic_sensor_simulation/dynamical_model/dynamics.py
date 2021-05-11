# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC
import logging
from atomic_sensor_simulation.operable_functions import create_operable_const_func, create_operable_cos_func, create_operable_sin_func


class DynamicalModel(ABC):
    """Class holding a matrix of functions - linear/ non-linear, discrete/continuous, time-dependent/time independent"""
    def __init__(self,
                 transition_matrix):
        """
        :param transition_matrix: matrix of functions
        """
        self.__transition_matrix = transition_matrix

    def evaluate_transition_matrix_at_time_t(self, time):
        """Evaluates the matrix of functions at time t"""
        matrix_flat = self.__transition_matrix.flatten()
        shape = np.shape(self.__transition_matrix)
        evaluated_matrix = np.empty_like(matrix_flat)
        for index, element in np.ndenumerate(matrix_flat):
            evaluated_matrix[index] = matrix_flat[index](time)
        return np.reshape(evaluated_matrix, shape)

    def step(self, state_mean, state, time, time_step):
        raise NotImplementedError('step function not implemented')


class LinearDifferentialDynamicalModel(DynamicalModel):

    def __init__(self, transition_matrix, logger=None):
        self.transition_matrix = transition_matrix
        self._logger = logger or logging.getLogger(__name__)
        DynamicalModel.__init__(self, transition_matrix, logger)

    def step(self, state_mean, state, time, time_step, intrinsic_noise=None):
        self._logger.debug('Performing a step for time %r' % str(time))
        state_mean.update(state_mean.vec + self.evaluate_transition_matrix_at_time_t(time).dot(
            state_mean.vec) * time_step)
        if intrinsic_noise:
            state.update(state_mean.vec + intrinsic_noise.step())
        else:
            state.update(state_mean)
        return

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
    