#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import ExtendedKalmanFilter
from scipy.integrate import odeint, simps
import sympy
import numpy as np
from atomic_sensor_simulation.filter_model.model import Model
from atomic_sensor_simulation.utilities import eval_matrix_of_functions


class Extended_KF(Model):

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
                 num_terms,
                 time_arr
                 ):

        Model.__init__(self,
                       Q=Q,
                       R=R,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
        self.F = F
        self.x0 = x0
        self.H = H
        self.P0 = P0
        self.num_terms = num_terms
        self.dim_x = len(self.x0)
        self.time_arr = time_arr

    def hx(self, x):
        return self.H.dot(x)

    def HJacobianat(self, x):
        return self.H

    def initialize_filterpy(self, **kwargs):
        self._logger.info('Initializing Extended Kalman Filter (filtepy)...')
        filterpy = AtomicSensorEKF(dim_x=self.dim_x,
                                   dim_z=self.dim_z,
                                   num_terms=self.num_terms,
                                   dt=self.dt,
                                   x0=self.x0,
                                   P0=self.P0,
                                   F=self.F,
                                   time_resolution_ode_solver=30,
                                   **kwargs)
        filterpy.x = self.x0
        filterpy.P = self.P0
        filterpy.Q = self.Q
        filterpy.R = self.R_delta

        return filterpy


class AtomicSensorEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, x0, P0, F, time_resolution_ode_solver,**kwargs):
        ExtendedKalmanFilter.__init__(self, dim_x, dim_z)
        self.dt = dt
        self.t = 0
        self.__light_correlation_const = kwargs['light_correlation_const']
        self.__coupling_amplitude = kwargs['coupling_amplitude']
        self.__coupling_freq = kwargs['coupling_freq']
        self.__coupling_phase_shift = kwargs['coupling_phase_shift']
        self.__larmour_freq = kwargs['larmour_freq']
        self.__spin_correlation_const = kwargs['spin_correlation_const']
        self.x0 = x0
        self.P0 = P0
        self.F = F
        self.time_resolution = time_resolution_ode_solver

    def set_fxu(self, fxu):
        self.fxu = fxu
        return

    def set_Q(self, Q):
        self.Q = Q
        return

    def predict(self, u=0):
        self.x = self.predict_x_ode_solve(from_time=self.t, time_resolution=self.time_resolution)
        self.P = self.predict_cov_ode_solve(from_time=self.t, time_resolution=self.time_resolution)
        self.t += self.dt
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def predict_x_ode_solve(self, from_time, time_resolution=30):
        x0 = np.array(np.reshape(self.x, 4), dtype=np.float64)
        def dxdt(x,t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self.F, t), dtype=float),
                                     np.reshape(x, (4, -1))), 4)

        t = np.linspace(from_time, from_time + self.dt, num=time_resolution)  # times to report solution
        # store solution
        x = None
        # solve ODE
        for i in range(1, time_resolution):
            # span for next time step
            tspan = [t[i - 1], t[i]]
            # solve for next step
            x = odeint(dxdt, x0, tspan)
            # next initial condition
            x0 = x[1]
        return np.reshape(x[1], (4, -1))

    def predict_cov_ode_solve(self, from_time, time_resolution=30):
        P0 = np.reshape(self.P,  16)
        def dPdt(P,t):
            result = np.dot(np.array(eval_matrix_of_functions(self.F, t), dtype=float), np.reshape(P, (4, 4))) +\
            np.dot(np.reshape(P, (4, 4)), np.transpose(np.array(eval_matrix_of_functions(self.F, t), dtype=float))) +\
                self.Q
            return np.reshape(result, 16)
        t = np.linspace(from_time, from_time + self.dt, num=time_resolution)  # times to report solution
        # store solution
        P = None
        # solve ODE
        for i in range(1, time_resolution):
            # span for next time step
            tspan = [t[i - 1], t[i]]
            # solve for next step
            P = odeint(dPdt, P0, tspan)
            # next initial condition
            P0 = P[1]
        return np.reshape(P[1], (4, 4))
