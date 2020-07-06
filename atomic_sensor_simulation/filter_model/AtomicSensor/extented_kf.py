#!/usr/bin/env python
# -*- coding: utf-8 -*-
from filterpy.kalman import ExtendedKalmanFilter
from scipy.integrate import odeint, simps
import numpy as np
import copy
from scipy.linalg import expm
from atomic_sensor_simulation.filter_model.model import Model
from atomic_sensor_simulation.utilities import eval_matrix_of_functions


def reshape_z(z, dim_z, ndim):
    """ ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z

class Extended_KF(Model):

    def __init__(self,
                 F,
                 Q,
                 H,
                 R,
                 R_delta,
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
                       R_delta=R_delta,
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
                                   H=self.H,
                                   R=self.R,
                                   R_delta=self.R_delta,
                                   time_resolution_ode_solver=10,
                                   **kwargs)
        filterpy.x = self.x0
        filterpy.P = self.P0
        filterpy.Q = self.Q
        filterpy.R = self.R

        return filterpy


class AtomicSensorEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, x0, P0, F, H, R, R_delta, time_resolution_ode_solver, **kwargs):
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
        self.H = H
        self.R = R
        self.R_delta = R_delta
        self.Phi_0 = np.identity(4)
        self.Phi = np.identity(4)
        self.Q_delta = self.compute_Q_delta(from_time=self.t, Phi_0=self.Phi, num_terms=30)
        self.time_resolution = time_resolution_ode_solver

    def set_fxu(self, fxu):
        self.fxu = fxu
        return

    def set_Q(self, Q):
        self.Q = Q
        return

    def compute_Q_delta(self, from_time, Phi_0, num_terms=30):
        def dPhidt(Phi, t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self.F, t), dtype=float),
                                     np.reshape(Phi, (4, 4))), 16)

        t = np.linspace(from_time, from_time + self.dt, num=num_terms)  # times to report solution
        Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, 16), t, full_output=True)
        # Numerical
        Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (4, 4)) for i in range(len(Phi_deltas))]
        Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
        integrands = np.array(
            [np.dot(np.dot(a, self.Q), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
        int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23, int30, int31, int32, int33 = map(
            list, zip(*integrands.reshape(*integrands.shape[:1], -1)))
        integrand_split = [int00, int01, int02, int03, int10, int11, int12, int13, int20, int21, int22, int23, int30,
                           int31, int32, int33]
        # calculate integral numerically using simpsons rule
        self.Q_delta = np.reshape(np.array([simps(i, t) for i in integrand_split]), (4, 4))
        return np.reshape(np.array([simps(i, t) for i in integrand_split]), (4, 4))

    def predict(self, u=0):
        self.x = self.predict_x_ode_solve(from_time=self.t, time_resolution=self.time_resolution)
        self.P = self.predict_cov_ode_solve(from_time=self.t, time_resolution=self.time_resolution)
        self.t += self.dt
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def predict_discretization_first(self, u=0):
        self.Phi = self.predict_Phi_odeint(from_time=self.t, time_resolution=self.time_resolution)
        self.Q_delta = self.compute_Q_delta(from_time=self.t, Phi_0=self.Phi, num_terms=self.time_resolution)
        self.x = np.dot(self.Phi, self.x)
        self.P = np.dot(np.dot(self.Phi, self.P), self.Phi.T) + self.Q_delta
        self.t += self.dt
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def predict_Phi_odeint(self, from_time, time_resolution=30):
        Phi0 = np.reshape(np.identity(4), 16)

        def dPhidt(Phi, t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self.F, t), dtype=float),
                                     np.reshape(Phi, (4, 4))), 16)

        t = np.linspace(from_time, from_time + self.dt, num=time_resolution)  # times to report solution
        # store solution
        Phi = None
        # solve ODE
        for i in range(1, time_resolution):
            # span for next time step
            tspan = [t[i - 1], t[i]]
            # solve for next step
            Phi = odeint(dPhidt, Phi0, tspan)
            # next initial condition
            Phi0 = Phi[1]
        return np.reshape(Phi[1], (4, 4))

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
                self.Q - np.dot(np.dot(np.dot(np.dot(np.reshape(P, (4, 4)), np.transpose(self.H)), np.linalg.inv(self.R)), self.H), np.reshape(P, (4, 4)))
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

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        z = reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R_delta
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        if Hx is None:
            pass
        H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x)

        # common subexpression for speed
        PHT = np.dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = np.linalg.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = copy.deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
