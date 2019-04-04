#!/usr/bin/env python
# -*- coding: utf-8 -*-
from atomic_sensor_simulation.utilities import stringify_namespace, load_logging_config
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.state import State
from atomic_sensor_simulation.atomic_sensor import AtomicSensor
from atomic_sensor_simulation import CONSTANTS
from filterpy.stats import plot_covariance
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter

import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging


def main():
    # setup a logger
    load_logging_config()
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of Atomic Sensor Simulation.')

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Atomic Sensor Simulation')
    # parser.add_argument('--working_dir', action='store', help='foo help')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "run-tests" command
    tests_parser = subparsers.add_parser('run-tests', help='Run unit tests')
    tests_parser.set_defaults(func=run_tests)

    # create the parser for the "run-simulation" command
    simulation_parser = subparsers.add_parser('run-simulation', help='Run atomic sensor simulation')
    simulation_parser.add_argument('-o',
                                   '--output_path',
                                   action='store',
                                   help='A string representing path where the output should be saved.',
                                   default='./')
    simulation_parser.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    simulation_parser.add_argument('--save_data_file',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    simulation_parser.set_defaults(func=run_simulation)

    # create the parser for the "run-wiener" command
    tests_parser = subparsers.add_parser('run-wiener', help='')
    tests_parser.set_defaults(func=run_wiener)

    # parse some argument lists
    args = parser.parse_args()
    logger.info('Parsed input arguments %r' % stringify_namespace(args))
    args.func(args)
    logger.info('Ending execution of the application.')
    return 0


def run_simulation(*args):
    pass

def run_wiener(*args):

    def linear_kalman():
        kalman_filter = KalmanFilter(dim_x=2, dim_z=1)
        kalman_filter.x = np.array([spin_initial_val, quadrature_initial_val])
        kalman_filter.F = np.array([[-atoms_correlation_const*dt, dt * CONSTANTS.g_a_COUPLING_CONST], [0, 1]])
        kalman_filter.H = np.array([[CONSTANTS.g_d_COUPLING_CONST, 0]])
        kalman_filter.P *= CONSTANTS.SCALAR_STREGTH_y
        kalman_filter.R = np.array([[CONSTANTS.SCALAR_STREGTH_y]])
        kalman_filter.B = np.array([[1,0],[0,0]])
        from filterpy.common import Q_discrete_white_noise
        # kalman_filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
        kalman_filter.Q = block_diag(CONSTANTS.SCALAR_STREGTH_j, CONSTANTS.SCALAR_STRENGTH_q)
        return kalman_filter

    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-wiener command.')

    # initial conditions
    spin_initial_val = 0.0
    quadrature_initial_val = 0.0
    dt = 1.
    num_iter = 200
    atoms_correlation_const = 0.000001
    omega=0.05
    amplitude = 0.

    state = State(spin=spin_initial_val,
                  quadrature=quadrature_initial_val,
                  noise_spin=GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STREGTH_j, dt=dt),
                  noise_quadrature=GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STRENGTH_q,
                                                      dt=dt),
                  dt=dt,
                  atoms_correlation_const=atoms_correlation_const,
                  omega=omega,
                  amplitude=amplitude)
    sensor = AtomicSensor(state, scalar_strenght_y=CONSTANTS.SCALAR_STREGTH_y, dt=dt)
    print(np.array(range(num_iter)))

    zs, qs = zip(*np.array(
        [(sensor.read(_)) for _ in range(num_iter)]))  # read photocurrent values from the sensor
    print(qs)
    # zs_minus_noise = np.subtract(zs, noise)
    # # plot (time, photocurrent) for values with noise and for values without noise
    # plt.plot(range(num_iter), zs, 'r', label='Noisy sensor detection')
    # # plt.plot(range(num_iter), noise, 'b', label='Noise')
    # plt.plot(range(num_iter), zs_minus_noise, 'g', label='Noisy sensor detection minus noise')
    # plt.plot(range(num_iter), zs_no_noise, 'k', label='Ideal data')
    # plt.legend()
    # plt.show()
    Fs = [np.array([[-atoms_correlation_const*_, _ * CONSTANTS.g_a_COUPLING_CONST], [0, 1]]) for _ in range(num_iter)]
    us = [np.array([amplitude*np.sin(omega*_), 0]).T for _ in range(num_iter)]

    kalman_filter = linear_kalman()
    (mu, cov, _, _) = kalman_filter.batch_filter(zs, Fs=Fs, us=us)
    # (xs, Ps, Ks, _) = kalman_filter.rts_smoother(mu, cov, Fs=Fs)
    filtered_signal = mu[:,1]

    # # plot results
    plt.plot(range(num_iter), qs, 'k')  # sensor readings
    # plt.plot(range(num_iter), np.ones(num_iter) * 0.5, 'b')  # sensor readings
    plt.plot(range(num_iter), filtered_signal, 'r')  # sensor readings
    # plt.ylim(0.4,0.6)
    plt.show()

def run_tests(*args):
    #:TODO implement this function
    pass


if __name__ == "__main__":

    main()
