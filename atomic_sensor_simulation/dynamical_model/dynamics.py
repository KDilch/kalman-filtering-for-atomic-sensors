# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC
import logging


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
        DynamicalModel.__init__(self, transition_matrix)

    def step(self, state_mean, state, time, time_step, intrinsic_noise=None):
        self._logger.debug('Performing a step for time %r' % str(time))
        state_mean.update(state_mean.vec + self.evaluate_transition_matrix_at_time_t(time).dot(
            state_mean.vec) * time_step)
        if intrinsic_noise:
            state.update(state_mean.vec + intrinsic_noise.step())
        else:
            state.update(state_mean)
        return


