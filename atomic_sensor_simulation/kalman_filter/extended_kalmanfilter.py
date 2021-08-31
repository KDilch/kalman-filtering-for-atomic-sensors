#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import abc
import copy
import logging


class ExtendedKalmanFilterStrategy(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'predict') and
                callable(subclass.predict) and
                hasattr(subclass, 'update') and
                callable(subclass.update) or
                NotImplemented)

    @abc.abstractmethod
    def predict(self, ekf):
        raise NotImplementedError("")

    @abc.abstractmethod
    def update(self, ekf):
        raise NotImplementedError("")


class ExtendedKFDiscretizationFirstStrategy(ExtendedKalmanFilterStrategy):
    def __init__(self):
        pass

    def predict(self, ekf):
        pass

    def update(self, ekf):
        pass


class ExtendedKFLinearizationFirstStrategy(ExtendedKalmanFilterStrategy):
    def __init__(self):
        pass

    def predict(self, ekf):
        F = ekf.F



class ExtendedKalmanFilter(object):
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
        self._dynamical_model = dynamical_model
        self._measurement_model = measurement_model
        self.R = self._measurement_model.noise_cov_delta

        if prior is None:
            self.x, self.P = self.compute_x0_and_P0(z0)
            self.__logger.info('Setting default values for x0 and P0...')
        else:
            self.x, self.P = prior

        self.x_prior = None
        self.P_prior = None
        self.x_post = None
        self.P_post = None
        self.S = None
        self.SI = None
        self.Identity = np.eye(self._dynamical_model.dynamics.state_vec_shape)
        self.t = initial_time
        self.y = None  # Residual between the measurement and prediction
        self.K = None
        self.z = None  # store last measurement outcome

    def predict(self):
        self._dynamical_model.dynamics.extended_num_compute_discrete_transition_and_noise_matrices(from_time=self.t)
        self.x = np.dot(self._dynamical_model.dynamics.discrete_transition_matrix, self._dynamical_model.vec)
        self.P = np.dot(np.dot(self._dynamical_model.dynamics.discrete_transition_matrix, self.P), self._dynamical_model.dynamics.discrete_transition_matrix_T) + self._dynamical_model.dynamics.discrete_intrinsic_noise
        self.t += self._measurement_model.dt
        # save prior
        self.x_prior = np.copy(self._dynamical_model.vec)
        self.P_prior = np.copy(self.P)
        return

    def update(self, z):
        "In our case the measurement is still linear"
        self.y = z - np.dot(self._measurement_model.H, self._dynamical_model.vec)  # y = z - Hx
        PHT = np.dot(self.P, self._measurement_model.H_T)
        self.S = np.dot(self._measurement_model.H, PHT) + self.R  # system uncertainty -> P projected to measurement space
        self.SI = np.linalg.inv(self.S)
        self.K = np.dot(PHT, self.SI)
        self._dynamical_model._state.update(self.x + np.dot(self.K, self.y))
        I_KH = self.Identity - np.dot(self.K, self._measurement_model.H)
        self.P = np.dot(np.dot(I_KH, self.P), np.transpose(I_KH)) + np.dot(np.dot(self.K, self.R), self.K.T)
        self.z = copy.deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        return

    def compute_x0_and_P0(self, z0):
        x0 = np.dot(self._measurement_model.H_inverse, z0)
        cov_x0 = self._dynamical_model.dynamics.discrete_intrinsic_noise + np.dot(np.dot(self._measurement_model.H_inverse, self.R),
                                                                                    np.transpose(self._measurement_model.H_inverse))
        return x0, cov_x0

    @property
    def discrete_transition_matrix(self):
        return self._dynamical_model.dynamics.discrete_transition_matrix

    @property
    def continuous_transition_matrix(self):
        return np.array(self._dynamical_model.dynamics.evaluate_transition_matrix_at_time_t(time=self.t), dtype=float)

    @property
    def discrete_intrinsic_noise_matrix(self):
        return self._dynamical_model.dynamics.discrete_intrinsic_noise

    @property
    def intrinsic_noise_matrix(self):
        return self._dynamical_model.dynamics.intrinsic_noise.cov

    @property
    def discrete_measurement_noise_matrix(self):
        return self.R

    @property
    def dt(self):
        return self._measurement_model.dt

    @property
    def measurement_matrix(self):
        return self._measurement_model.H

    @property
    def state_vec_shape(self):
        return self._dynamical_model.dynamics.state_vec_shape
