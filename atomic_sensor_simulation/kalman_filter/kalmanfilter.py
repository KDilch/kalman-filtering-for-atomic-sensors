#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import logging


class KalmanFilter(object):
    """A class that is able to perform Kalman filtering for continuous, discrete as well as hybrid dynamical
     and measurement models. For now I will only address the hybrid continuous dynamical model and discrete measurement
      model."""

    def __init__(self, dynamical_model, measurement_model, prior=None, z0=None, initial_time=0, logger=None):
        """
        :param dynamical_model: object that has information about the system dynamics and well as intrinsic noise
        :param measurement_model: object that has the information about the measurement performed on the system as well as a measurement noise
        :param prior: tuple (x0, P0) - mean and covariance of the prior distribution
        """
        self.__logger = logger if logger else logging.getLogger(__name__)
        self.__dynamical_model = dynamical_model
        self.__measurement_model = measurement_model

        if any(prior[0]) is None or prior[1] is None:
            if z0:
                self.x0, self.P0 = self.compute_x0_and_P0(z0)
                self.__logger.info('Setting default values for x0 and P0...')
            else:
                raise ValueError('')
        else:
            self.x0, self.P0 = prior

        self.x_prior = None
        self.P_prior = None
        self.x_post = None
        self.P_post = None
        self.P = None
        self.x = None
        self.t = initial_time

    def predict(self):
        self.x = np.dot(self.__dynamical_model.num_discrete_transition_matrix, self.x)
        self.P = np.dot(np.dot(self.__dynamical_model.num_discrete_transition_matrix, self.P), self.__transpose_dynamical_model.num_discrete_transition_matrix) + self.__dynamical_model.num_discrete_noise_matrix
        self.t += self.__measurement_model.dt
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self):
        pass

    def compute_x0_and_P0(self, z0):
        x0 = np.dot(self.__measurement_model.H_inverse, z0)
        cov_x0 = self.__dynamical_model.num_discrete_noise_matrix + np.dot(np.dot(self.__measurement_model.H_inverse, self.__measurement_model.R_delta), np.transpose(self.__measurement_model.H_inverse))
        return x0, cov_x0
