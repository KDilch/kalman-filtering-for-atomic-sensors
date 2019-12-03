#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from scipy.integrate import odeint

from atomic_sensor_simulation.filter_model.model import Model
from atomic_sensor_simulation.utilities import integrate_matrix_of_functions, eval_matrix_of_functions


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

        self.Phi_delta = self.compute_Phi_delta_exp_Fdt_approx(from_time=0)
        self.Q_delta = np.dot(np.dot(self.Phi_delta, self.Q), self.Phi_delta.transpose()) * dt
        if any(x0) is None or P0 is None:
            self.x0, self.P0 = self.calculate_x0_and_P0(z0)
            self._logger.info('Setting default values for x0 and P0...')
        else:
            self.x0 = x0
            self.P0 = P0
        self.dim_x = len(self.x0)


    def compute_Phi_delta_exp_Fdt_approx(self, from_time):
        """Returns solution for Phi from t_k to t_k+dt_filter as if F did not depend on time. Can be used for very slowly varying functions etc.
        :param from_time: start time of the current step (t_k)
        :return: exp(F(t)dt)
        """
        return expm(eval_matrix_of_functions(self._F, from_time)*self.dt)

    def compute_Phi_delta_exp_int_approx(self, from_time):
        """Returns solution for Phi t_k to t_k+dt_filter but the time-ordering operator is not taken into account.
        :param from_time: start time of the current step (t_k)
        :return: exp(integral F(t)dt)
        """
        return expm(integrate_matrix_of_functions(self._F, from_time, from_time + self.dt))

    def compute_Phi_delta_solve_ode_numerically(self, from_time, Phi_0, time_resolution=10):
        """
        :param from_time: start time of the current step (t_k)
        :param Phi_0: initial condition Phi from t_k-dt_filter to t_k
        :param time_resolution: Indicates for how many subsections time from t_k to t_k + dt_filter should be divided
        :return: numerical solution to differential equation: dPhi/dt=F(t)Phi
        """
        def dPhidt(Phi, t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self._F, t), dtype=float),
                                     np.reshape(Phi, (4, 4))), 16)

        t = np.linspace(from_time, from_time + self.dt, num=time_resolution)  # times to report solution
        Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, 16), t, full_output=True)
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

    def calculate_x0_and_P0(self, z0):
        x0 = np.dot(self.H_inverse, z0)
        print(x0)
        cov_x0 = self.Q + np.dot(np.dot(self.H_inverse, self.R_delta), np.transpose(self.H_inverse))
        print(cov_x0)
        return x0, cov_x0
