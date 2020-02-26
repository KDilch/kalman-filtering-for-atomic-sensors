#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from scipy.integrate import simps
from sympy import *
from scipy.linalg import expm, solve_discrete_are
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
        self.Q_delta = self.compute_Q_delta_sympy(from_time=0., Phi_0=self.Phi_delta)
        if any(x0) is None or P0 is None:
            self.x0, self.P0 = self.calculate_x0_and_P0(z0)
            self._logger.info('Setting default values for x0 and P0...')
        else:
            self.x0 = x0
            self.P0 = P0
        self.dim_x = len(self.x0)

    def compute_Q_delta_sympy(self, from_time, Phi_0, num_terms=30):
        def dPhidt(Phi, t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self._F, t), dtype=float),
                                     np.reshape(Phi, (4, 4))), 16)

        t = np.linspace(from_time, from_time + self.dt, num=num_terms)  # times to report solution
        Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, 16), t, full_output=True)
        #NUMerical
        Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (4, 4)) for i in range(len(Phi_deltas))]
        Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
        integrands = np.array([np.dot(np.dot(a, self.Q), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
        #Assuming 4x4 matrices! #TODO get rid of it
        a = integrands.reshape(-1, integrands.shape[-1])
        int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23, int30, int31, int32, int33 = map(list, zip(*integrands.reshape(*integrands.shape[:1], -1)))
        integrand_split = [int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23, int30, int31, int32, int33]
        #calculate integral numerically using simpsons rule
        self.Q_delta = np.reshape(np.array([simps(i, t) for i in integrand_split]), (4, 4))
        return np.reshape(np.array([simps(i, t) for i in integrand_split]), (4, 4))
        # #SYMPY
        # out = zeros(*(self.Phi_delta.shape))
        # for n in range(num_terms):
        #     matrix_to_n = matrix_power(self.Phi_delta, n) / factorial(n)
        #     Phi_Q_Phi_t_Nth_term = np.dot(np.dot(matrix_to_n, self.Q), np.transpose(matrix_to_n))
        #     matrix_flat = Phi_Q_Phi_t_Nth_term.flatten()
        #     shape = np.shape(Phi_Q_Phi_t_Nth_term)
        #     matrix = np.empty_like(matrix_flat)
        #     for index, element in np.ndenumerate(matrix_flat):
        #         matrix[index] = lambda t: t ** (2 * n) * element
        #     Phi_Q_PHI_T = np.reshape(matrix, shape)
        #     from utilities import integrate_matrix_of_functions
        #     int = integrate_matrix_of_functions(Phi_Q_PHI_T, from_x=from_time, to_x=from_time+self.dt)
        #     Phi_delta_RF = expm(self.Phi_delta * self.dt)
        #     out += np.dot(np.dot(Phi_delta_RF, int), np.transpose(Phi_delta_RF))
        #     print(np.array(out).astype(np.float64))
        # return np.array(out).astype(np.float64)

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

    def compute_Phi_delta_solve_ode_numerically(self, from_time, Phi_0, time_resolution=30):
        """
        :param from_time: start time of the current step (t_k)
        :param Phi_0: initial condition Phi from t_k-dt_filter to t_k
        :param time_resolution: Indicates for how many subsections time from t_k to t_k + dt_filter should be divided
        :return: numerical solution to differential equation: dPhi/dt=F(t)Phi
        """
        Phi_0 = np.reshape(np.identity(4), 16)
        def dPhidt(Phi, t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self._F, t), dtype=float),
                                     np.reshape(Phi, (4, 4))), 16)

        t = np.linspace(from_time, from_time + self.dt, num=time_resolution)  # times to report solution
        # store solution
        x = np.empty_like(t)
        y = np.empty_like(t)
        Phi=None
        # solve ODE
        for i in range(1, time_resolution):
            # span for next time step
            tspan = [t[i - 1], t[i]]
            # solve for next step
            Phi = odeint(dPhidt, Phi_0, tspan)
            # store solution for plotting
            x[i] = Phi[1][0]
            y[i] = Phi[1][1]
            # next initial condition
            Phi_0 = Phi[1]
        # Phi_deltas = odeint(dPhidt, np.reshape(Phi_0, 16), t)
        # plt.scatter(t, x)
        # plt.xlabel('time')
        # plt.ylabel('x(t)')
        # plt.show()
        return np.reshape(Phi[1], (4, 4))

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
