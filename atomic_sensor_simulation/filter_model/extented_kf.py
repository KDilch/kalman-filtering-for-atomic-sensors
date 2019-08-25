#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import ExtendedKalmanFilter
from sympy import *
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
                 P0
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
        self.dim_x = len(self.x0)

    def initialize_filterpy(self):
        self._logger.info('Initializing Extended Kalman Filter (filtepy)...')
        filterpy = ExtendedKalmanFilter(dim_x=self.dim_x,
                                         dim_z=self.dim_z)
        filterpy.x = self.x0.T
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
    def __init__(self, dim_x, dim_z, num_terms):
        ExtendedKalmanFilter.__init__(self, dim_x, dim_z)
        jy, jz, q, p, dt, t = symbols('jy, jz, q, p, dt, t')
        self.fxu = Matrix([[]])
        F = Matrix([-])

            np.array([[create_operable_const_func(-self.__spin_correlation_const),
                                         create_operable_const_func(self.__larmour_freq),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0)],

                                        [create_operable_const_func(-self.__larmour_freq),
                                          create_operable_const_func(-self.__spin_correlation_const),
                                          create_operable_cos_func(amplitude=self.__coupling_amplitude,
                                                                   omega=self.__coupling_freq,
                                                                   phase_shift=self.__coupling_phase_shift),
                                          create_operable_sin_func(amplitude=self.__coupling_amplitude,
                                                                   omega=self.__coupling_freq,
                                                                   phase_shift=self.__coupling_phase_shift)],

                                        [create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(-self.__light_correlation_const),
                                         create_operable_const_func(0)
                                         ],

                                        [create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(0),
                                         create_operable_const_func(-self.__light_correlation_const)]])
        self.F_j = compute_expm_approx(Matrix(F*t), num_terms)
        self.subs = {jy: 0, jz: 0, q: 0, p: 0, t: 0, dt: dt}
        self.jy, self.jz, self.q, self.p = jy, jz, q, p
        self.dt = dt
        self.t = t

    def predict(self, u=0):
        self.x = self.move(self.x, u, self.dt)

        self.subs[self.jy] = self.x[0]
        self.subs[self.jz] = self.x[1]
        self.subs[self.q] = self.x[2]
        self.subs[self.p] = self.x[3]
        self.F_j

        F = array(self.F_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = array([[self.std_vel*u[0]**2, 0],
                   [0, self.std_steer**2]])

        self.P = dot(F, self.P).dot(F.T) + dot(V, M).dot(V.T)

    def move(self, x, u, dt):
        pass

def compute_expm_approx(matrix, num_terms):
    out = zeros(*(matrix.shape))
    for n in range(num_terms):
        out += matrix ** n / factorial(n)
    return out


