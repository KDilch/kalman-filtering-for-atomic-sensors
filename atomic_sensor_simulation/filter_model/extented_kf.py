#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import ExtendedKalmanFilter
from sympy import *
import sympy
import numpy as np
from scipy.linalg import expm

from atomic_sensor_simulation.filter_model.model import Model
from atomic_sensor_simulation.utilities import eval_matrix_of_functions


class Extended_KF(Model):

    def __init__(self,
                 F,
                 Q,
                 hx,
                 R,
                 Gamma,
                 u,
                 z0,
                 dt,
                 x0,
                 P0,
                 num_terms
                 ):

        Model.__init__(self,
                       Q=Q,
                       R=R,
                       Gamma=Gamma,
                       u=u,
                       z0=z0,
                       dt=dt)
        self.F = F
        self.Phi = expm(eval_matrix_of_functions(self.F, 0)*self.dt)
        self.hx = hx
        self.x0 = x0
        self.P0 = P0
        self.num_terms = num_terms
        self.dim_x = len(self.x0)

    def initialize_filterpy(self, **kwargs):
        self._logger.info('Initializing Extended Kalman Filter (filtepy)...')
        filterpy = AtomicSensorEKF(dim_x=self.dim_x,
                                   dim_z=self.dim_z,
                                   num_terms=self.num_terms,
                                   dt=self.dt,
                                   **kwargs)
        filterpy.x = self.x0
        filterpy.P = self.P0
        filterpy.Q = self.Q
        filterpy.R = self.R_delta

        return filterpy

    def compute_F_extended(self):
        #TODO sympy version
        jy = symbols('jy')
        jz = symbols('jz')
        q = symbols('q')
        p = symbols('p')
        x = Matrix(np.array([jy, jz, q, p]).T)
        F_ext = Matrix(eval_matrix_of_functions(self.F, 0)*x + x)
        print('F_ext', F_ext.jacobian(x))
        return F_ext


class AtomicSensorEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, num_terms, dt, **kwargs):
        ExtendedKalmanFilter.__init__(self, dim_x, dim_z)
        self.dt = dt
        self.t = 0

        self.__light_correlation_const = kwargs['light_correlation_const']
        self.__coupling_amplitude = kwargs['coupling_amplitude']
        self.__coupling_freq = kwargs['coupling_freq']
        self.__coupling_phase_shift = kwargs['coupling_phase_shift']
        self.__larmour_freq = kwargs['larmour_freq']
        self.__spin_correlation_const = kwargs['spin_correlation_const']
        jy, jz, q, p, deltat, time = symbols('jy, jz, q, p, deltat, time')
        self.fxu = Matrix([[]])
        self.num_terms = num_terms
        A = Matrix([[-self.__spin_correlation_const,
                                         self.__larmour_freq,
                                         0,
                                         0],

                                        [-self.__larmour_freq,
                                          -self.__spin_correlation_const,
                                          self.__coupling_amplitude*sympy.cos(self.__coupling_freq*time + self.__coupling_phase_shift),
                                          self.__coupling_amplitude*sympy.sin(self.__coupling_freq*time + self.__coupling_phase_shift)],

                                        [0,
                                         0,
                                         -self.__light_correlation_const,
                                         0
                                         ],

                                        [0,
                                         0,
                                         0,
                                         -self.__light_correlation_const]])

        self.A = A
        self.F_j = compute_expm_approx(Matrix(A*time), num_terms)
        self.subs = {jy: 0, jz: 0, q: 0, p: 0, time: 0, deltat: self.dt}
        self.jy, self.jz, self.q, self.p = jy, jz, q, p
        self.time = time

    def predict(self, u=0):
        self.t += self.dt

        self.subs[self.jy] = self.x[0]
        self.subs[self.jz] = self.x[1]
        self.subs[self.q] = self.x[2]
        self.subs[self.p] = self.x[3]
        self.F_j = compute_expm_approx(Matrix(self.A*self.t), self.num_terms)
        self.subs[self.time] = self.t

        self.x = self.move(self.x, u, self.dt)

        F = np.array(self.F_j.evalf(subs=self.subs)).astype(float)

        self.P = np.dot(F, self.P).dot(F.T)

    def move(self, x, u, dt):
        A = self.A.evalf(self.subs)
        return x + A.dot(x) * dt

def compute_expm_approx(matrix, num_terms):
    out = zeros(*(matrix.shape))
    for n in range(num_terms):
        out += matrix ** n / factorial(n)
    return out


