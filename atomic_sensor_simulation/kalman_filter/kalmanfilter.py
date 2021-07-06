#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
import logging


class DD_KalmanFilter(object):
    """A class that is able to perform Discrete-Discrete Kalman filtering for continuous, discrete as well as hybrid dynamical
     and measurement models. For now I will only address the hybrid continuous dynamical model and discrete measurement
      model."""

    def __init__(self, dynamical_model, measurement_model, z0, prior=None, initial_time=0, logger=None):
        """
        :param dynamical_model: object that has information about the system dynamics and well as intrinsic noise
        :param measurement_model: object that has the information about the measurement performed on the system as well as a measurement noise
        :param prior: tuple (x0, P0) - mean and covariance of the prior distribution
        """
        self.__logger = logger if logger else logging.getLogger(__name__)
        self.__dynamical_model = dynamical_model
        self.__measurement_model = measurement_model
        self.R = self.__measurement_model.noise_cov_delta(self.__dynamical_model._dynamics.discrete_dt)


        if prior is None:
            self.x0, self.P0 = self.compute_x0_and_P0(z0)
            self.__logger.info('Setting default values for x0 and P0...')
        else:
            self.x0, self.P0 = prior

        self.x_prior = None
        self.P_prior = None
        self.x_post = None
        self.P_post = None
        self.P = None
        self.S = None
        self.SI = None
        self.Identity = np.eye(self.__dynamical_model._dynamics.state_vec_shape)
        self.t = initial_time
        self.y = None  # Residual between the measurement and prediction
        self.K = None
        self.z = None  # store last measurement outcome

    def predict(self):
        self.x = np.dot(self.__dynamical_model.num_discrete_transition_matrix, self.__dynamical_model.state.vec)
        self.P = np.dot(np.dot(self.__dynamical_model.__discrete_transition_matrix, self.P), self.__dynamical_model.__discrete_transition_matrix_T) + self.__dynamical_model.discrete_noise_matrix
        self.t += self.__measurement_model.dt
        # save prior
        self.x_prior = np.copy(self.__dynamical_model.state.vec)
        self.P_prior = np.copy(self.P)
        return

    def update(self, z):
        self.y = z - np.dot(self.__measurement_model.H, self.__dynamical_model.state.vec)  # y = z - Hx
        PHT = np.dot(self.P, self.__measurement_model.H_T)
        self.S = np.dot(self.__measurement_model.H_T, PHT) + self.__measurement_model.R_delta  # system uncertainty -> P projected to measurement space
        self.SI = np.linalg.inv(self.S)
        self.K = np.dot(PHT, self.SI)
        self.__dynamical_model.state.vec.assign_new_vec(self.x + np.dot(self.K, self.y))
        I_KH = self.Identity - np.dot(self.K, self.__measurement_model.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, self.__measurement_model.R_delta), self.K.T)
        self.z = copy.deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        return

    def compute_x0_and_P0(self, z0):
        x0 = np.dot(self.__measurement_model.H_inverse, z0)
        cov_x0 = self.__dynamical_model._dynamics.discrete_intrinsic_noise + np.dot(np.dot(self.__measurement_model.H_inverse, self.R),
                                                                                    np.transpose(self.__measurement_model.H_inverse))
        return x0, cov_x0
