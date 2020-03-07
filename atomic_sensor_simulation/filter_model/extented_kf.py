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
                                   **kwargs)
        filterpy.x = self.x0
        filterpy.P = self.P0
        filterpy.Q = self.Q
        filterpy.R = self.R_delta

        return filterpy


class AtomicSensorEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, x0, P0, **kwargs):
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

        jy, jz, q, p, deltat, time = sympy.symbols('jy, jz, q, p, deltat, time')
        self.x = sympy.Matrix([[jy], [jz], [q], [p]])

        self.F_sympy = sympy.Matrix([[-self.__spin_correlation_const, self.__larmour_freq, 0, 0],
                    [-self.__larmour_freq,
                     -self.__spin_correlation_const,
                     self.__coupling_amplitude*sympy.cos(self.__coupling_freq*time + self.__coupling_phase_shift),
                     self.__coupling_amplitude*sympy.sin(self.__coupling_freq*time + self.__coupling_phase_shift)],
                    [0, 0, -self.__light_correlation_const, 0],
                    [0, 0, 0, -self.__light_correlation_const]])
        self.fxu = self.x + self.F_sympy*self.x*deltat
        self.fJacobian_at_x = self.fxu.jacobian(sympy.Matrix([jy, jz, q, p]))
        self.subs = {jy: self.x0[0], jz: self.x0[1], q: self.x0[2], p: self.x0[3], time: 0, deltat: dt}
        self.jy, self.jz, self.q, self.p = jy, jz, q, p
        self.time = time

    def set_F(self, F):
        self.F = F
        return

    def set_Q(self, Q):
        self.Q = Q
        return

    def predict(self, u=0):
        self.x = self.move()
        self.t += self.dt
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def move(self):
        self.x = np.dot(self.F, self.x)
        return self.x
        # fxu_current = self.x
        # for i in np.linspace(self.t, self.t+self.dt, num=1, endpoint=True):
        #     smaller_dt = self.dt
        #     fxu_current = self.x + self.F_sympy*self.x*smaller_dt
        #     self.x = fxu_current.evalf(subs=self.subs)
        #     self.subs[self.jy] = self.x[0]
        #     self.subs[self.jz] = self.x[1]
        #     self.subs[self.q] = self.x[2]
        #     self.subs[self.p] = self.x[3]
        #     self.subs[self.time] = self.t
        # return fxu_current.evalf(subs=self.subs)
