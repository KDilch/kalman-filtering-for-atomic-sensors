#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from scipy.integrate import odeint


from atomic_sensor_simulation.filter_model.model import Model
from atomic_sensor_simulation.utilities import integrate_matrix_of_functions, eval_matrix_of_functions
from atomic_sensor_simulation.homemade_kalman_filter.homemade_kf import HomeMadeKalmanFilter


class Linear_KF(Model):

    def __init__(self,
                 F,
                 Q,
                 H,
                 R,
                 Gamma,
                 u,
                 z0,
                 dt,
                 x0,
                 P0,
                 x_jac=None,
                 **kwargs
                 ):
        self._F = F
        self._x_jac=None
        self.Q = Q
        self.H = H
        self.H_inverse = np.linalg.pinv(self.H)
        self.__light_correlation_const = kwargs['light_correlation_const']
        self.__coupling_amplitude = kwargs['coupling_amplitude']
        self.__coupling_freq = kwargs['coupling_freq']
        self.__coupling_phase_shift = kwargs['coupling_phase_shift']
        self.__larmour_freq = kwargs['larmour_freq']
        self.__spin_correlation_const = kwargs['spin_correlation_const']

        Model.__init__(self,
                       Q=Q,
                       R=R,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)

        self.Phi_delta = self.compute_Phi_delta(from_time=0)
        self.Q_delta = np.dot(np.dot(self.Phi_delta, self.Q), self.Phi_delta.transpose()) * dt
        if any(x0) is None or P0 is None:
            self.x0, self.P0 = self.calculate_x0_and_P0(z0)
            self._logger.info('Setting default values for x0 and P0...')
        else:
            self.x0 = x0
            self.P0 = P0
        self.dim_x = len(self.x0)


    def compute_Phi_delta(self, from_time):
        return expm(eval_matrix_of_functions(self._F, from_time)*self.dt)
        # return expm(integrate_matrix_of_functions(self._F, from_time, from_time + self.dt))

    def compute_Phi_delta_odeint(self, from_time, Phi_0, time_resolution=10):
        def dPhidt(Phi, t):
            Phi_00, Phi_01, Phi_02, Phi_03, Phi_10, Phi_11, Phi_12, Phi_13, Phi_20, Phi_21, Phi_22, Phi_23, Phi_30, Phi_31, Phi_32, Phi_33 = np.reshape(Phi, 16)
            return np.array([-self.__spin_correlation_const*Phi_00+self.__larmour_freq*Phi_10,
                            -self.__spin_correlation_const*Phi_01+self.__larmour_freq*Phi_11,
                            -self.__spin_correlation_const*Phi_02+self.__larmour_freq*Phi_12,
                            -self.__spin_correlation_const*Phi_03+self.__larmour_freq*Phi_13,
                            -self.__larmour_freq*Phi_00 - self.__spin_correlation_const*Phi_10+
                             self.__coupling_amplitude*np.cos(self.__coupling_freq*t + self.__coupling_phase_shift)*Phi_20+
                             self.__coupling_amplitude*np.sin(self.__coupling_freq*t + self.__coupling_phase_shift)*Phi_30,
                            -self.__larmour_freq * Phi_01 - self.__spin_correlation_const * Phi_11+
                                self.__coupling_amplitude * np.cos(
                                self.__coupling_freq * t + self.__coupling_phase_shift) * Phi_21 +
                            self.__coupling_amplitude * np.sin(
                                  self.__coupling_freq * t + self.__coupling_phase_shift) * Phi_31,
                            -self.__larmour_freq * Phi_02 - self.__spin_correlation_const * Phi_12+
                            self.__coupling_amplitude * np.cos(
                                  self.__coupling_freq * t + self.__coupling_phase_shift) * Phi_22 +
                            self.__coupling_amplitude * np.sin(
                                  self.__coupling_freq * t + self.__coupling_phase_shift) * Phi_32,
                            -self.__larmour_freq * Phi_03 - self.__spin_correlation_const * Phi_13+
                            self.__coupling_amplitude * np.cos(
                                  self.__coupling_freq * t + self.__coupling_phase_shift) * Phi_23 +
                            self.__coupling_amplitude * np.sin(
                                  self.__coupling_freq * t + self.__coupling_phase_shift) * Phi_33,
                            -self.__light_correlation_const*Phi_20,
                            -self.__light_correlation_const*Phi_21,
                            -self.__light_correlation_const*Phi_22,
                            -self.__light_correlation_const*Phi_23,
                            -self.__light_correlation_const*Phi_30,
                            -self.__light_correlation_const*Phi_31,
                            -self.__light_correlation_const*Phi_32,
                            -self.__light_correlation_const*Phi_33])

        t = np.linspace(from_time, from_time + self.dt, num=time_resolution)  # times to report solution
        Phi_deltas, infodict = odeint(dPhidt, np.reshape(Phi_0, 16), t, full_output=True)
        return np.reshape(Phi_deltas[0], (4, 4))

    def initialize_filterpy(self):
        self._logger.info('Initializing Linear Kalman Filter (filtepy)...')
        filterpy = KalmanFilter(dim_x=len(self.x0), dim_z=self.dim_z)
        filterpy.x = self.x0
        filterpy.P = self.P0

        filterpy.F = self.Phi_delta
        filterpy.Q = self.Q_delta
        filterpy.H = self.H
        filterpy.R = self.R_delta
        return filterpy

    def initialize_homemade_filter(self):
        return HomeMadeKalmanFilter(x0=self.x0,
                                    P0=self.P0,
                                    Phi_delta=self.Phi_delta,
                                    Q_delta=self.Q_delta,
                                    H=self.H,
                                    R_delta=self.R_delta,
                                    Gamma=self.Gamma_control_transition_matrix,
                                    u=self.u_control_vec)

    def calculate_x0_and_P0(self, z0):
        x0 = np.dot(self.H_inverse, z0)
        print(x0)
        cov_x0 = self.Q + np.dot(np.dot(self.H_inverse, self.R_delta), np.transpose(self.H_inverse))
        print(cov_x0)
        return x0, cov_x0
