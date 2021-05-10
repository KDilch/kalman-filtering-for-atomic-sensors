# -*- coding: utf-8 -*-
from abc import ABC
import logging
from scipy.linalg import expm
from utilities import eval_matrix_of_functions


class Dynamics(ABC):
    """An abstract class representing any dynamical model (continuous/discrete, linear/non-linear)."""

    def __init__(self,
                 initial_vec,
                 noise,
                 coordinates_enum,
                 initial_time,
                 logger=None):

        self._logger = logger or logging.getLogger(__name__)
        self._state_vec = initial_vec
        self._mean_state_vec = initial_vec
        self._noise = noise
        self._coordinates = coordinates_enum
        self._time = initial_time

    def get_Phi_delta(self, method):
        """
        Phi is a transition matrix x_{k+1} = \Phi x_k [after linearization and discretization].
        :param method: please specify a method of computation - these are specified for particular implementations of dynamics
        :return:
        """
        raise NotImplemented("Phi_delta getter is not implemented.")

    def get_jacobian(self):
        raise NotImplemented("Jacobian getter is not implemented.")

    def step(self, t):
        raise NotImplemented("step function is not implemented.")


class LinearDifferentialContinuousModel(Dynamics):
    """An class representing any differential linear continuous dynamical model."""

    def __init__(self,
                 initial_vec,
                 noise,
                 coordinates_enum,
                 initial_time,
                 dt,
                 logger=None):
        """
        :param initial_vec:
        :param noise:
        :param coordinates_enum:
        :param initial_time:
        :param dt: note that it is a continues model so dt should be set to be appropriately small
        :param logger:
        """

        self._dt = dt
        self._F = None  # a place holder for a particular dynamical model
        Dynamics.__init__(self,
                          initial_vec,
                          noise,
                          coordinates_enum,
                          initial_time,
                          logger)

    def get_Phi_delta(self, method):
        return

    def get_jacobian(self):
        return

    def get_steady_state_solution(self):
        return

    def step(self, t):
        return

    def __compute_Phi(self):
        pass

    def compute_Phi_delta_time_invariant_approx(self):
        """Returns solution for Phi_delta from t_k to t_k+dt as if F did not depend on time -> (Phi=exp(F*delta_t)).
        Gives reasonable results for for very slowly varying functions etc.
        :return: exp(F(t)dt)
        """
        return expm(eval_matrix_of_functions(self._F, self._time) * self._dt)

    def compute_Phi_delta_exp_int_approx(self, from_time):
        """Returns solution for Phi t_k to t_k+dt_filter but the time-ordering operator is not taken into account.
        :param from_time: start time of the current step (t_k)
        :return: exp(integral F(t)dt)
        """
        return expm(integrate_matrix_of_functions(self._F, from_time, from_time + self.dt))

    def compute_Phi_delta_solve_ode_numerically(self, from_time, time_resolution=30):
        """
        :param from_time: start time of the current step (t_k)
        :param time_resolution: Indicates for how many subsections time from t_k to t_k + dt_filter should be divided
        :return: numerical solution to differential equation: dPhi/dt=F(t)Phi
        """
        Phi_0 = np.reshape(np.identity(4), 16)

        def dPhidt(Phi, t):
            return np.reshape(np.dot(np.array(eval_matrix_of_functions(self._F, t), dtype=float),
                                     np.reshape(Phi, (4, 4))), 16)

        t = np.linspace(from_time, from_time + self.dt, num=time_resolution)  # times to report solution
        # store solution
        Phi = None
        # solve ODE
        for i in range(1, time_resolution):
            # span for next time step
            tspan = [t[i - 1], t[i]]
            # solve for next step
            Phi = odeint(dPhidt, Phi_0, tspan)
            # next initial condition
            Phi_0 = Phi[1]
        return np.reshape(Phi[1], (4, 4))