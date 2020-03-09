#!/usr/bin/env python3
import unittest
import numpy as np
from filter_model.AtomicSensor.linear_kf import Linear_KF
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.state.atomic_state import AtomicSensorState


class TestNumericalSolutionForPhi(unittest.TestCase):

    def test_numerical_sol_for_Phi_time_independent(self):
        Q = np.array([[0.01, 0., 0., 0.],
                      [0., 0.01, 0., 0.],
                      [0., 0., 0.001, 0.],
                      [0., 0., 0., 0.001]])
        H = np.array([[0., 0.01, 0., 0.]])
        R = np.array([[0.01]])
        state = AtomicSensorState(initial_vec=np.array([2., 2., 0., 0.]),
                                  noise_vec=GaussianWhiteNoise(mean=[0., 0., 0., 0.],
                                                               cov=Q,
                                                               dt=0.005),
                                  initial_time=0,
                                  dt=0.005,
                                  light_correlation_const=0.33,
                                  spin_correlation_const=0.33,
                                  larmour_freq=6.,
                                  coupling_amplitude=50.,
                                  coupling_freq=0.,
                                  coupling_phase_shift=0.)
        fil = Linear_KF(F=state.F_transition_matrix,
                                Q=Q,
                                H=H,
                                R=R / 0.02,
                                Gamma=state.Gamma_control_evolution_matrix,
                                u=state.u_control_vec,
                                z0=[198.7521958574962],
                                dt=0.02,
                                x0=np.array([2., 2., 0., 0.]),
                                P0=None,
                                light_correlation_const=0.33,
                                spin_correlation_const=0.33,
                                larmour_freq=6.,
                                coupling_amplitude=50.,
                                coupling_freq=0.,
                                coupling_phase_shift=0.)
        linear_kf_filterpy = fil.initialize_filterpy()
        linear_kf_filterpy.predict()
        Phi_numerical = fil.compute_Phi_delta_solve_ode_numerically(from_time=0, Phi_0=linear_kf_filterpy.F)
        Phi_exact = fil.compute_Phi_delta_exp_Fdt_approx(from_time=0)
        self.assertAlmostEqual(Phi_numerical.all(), Phi_exact.all(), delta=0.001)
