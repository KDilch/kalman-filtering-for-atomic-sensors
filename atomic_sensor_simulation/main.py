#!/usr/bin/env python
# -*- coding: utf-8 -*-
from atomic_sensor_simulation.utilities import stringify_namespace, load_logging_config
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.state import AtomicSensorState
from atomic_sensor_simulation.atomic_sensor import AtomicSensor
from atomic_sensor_simulation import CONSTANTS
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

    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-wiener command.')

    # initial conditions
    spin_initial_val = 0.0
    quadrature_initial_val = 0.0
    dt = 1.
    num_iter = 200
    atoms_correlation_const = 0.000001
    omega = 0.1
    amplitude = 1.

    state = AtomicSensorState(initial_vec = np.array([spin_initial_val, quadrature_initial_val]),
                              noise_vec = np.array([GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STREGTH_j, dt=dt), GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STRENGTH_q, dt=dt)]),
                              evolution_matrix=1,
                              initial_time=0,
                              atoms_wiener_const=atoms_correlation_const,
                              g_a_coupling_const=CONSTANTS.g_a_COUPLING_CONST)

    sensor = AtomicSensor(state, scalar_strenght_y=CONSTANTS.SCALAR_STREGTH_y, dt=dt)

    zs, qs = zip(*np.array(
        [(sensor.read(_)) for _ in range(num_iter)]))  # read photocurrent values from the sensor

    from atomic_sensor_simulation.operable_functions import create_multiplicable_const_func, create_multiplicable_sin_func, create_multiplicable_cos_func
    F = np.exp(np.array([[-atoms_correlation_const*dt, dt * CONSTANTS.g_a_COUPLING_CONST], [0, dt]]))
    d_transition_matrix = np.array([[create_multiplicable_const_func(-atoms_correlation_const), create_multiplicable_const_func(CONSTANTS.g_a_COUPLING_CONST)], [create_multiplicable_const_func(0), create_multiplicable_const_func(1.)]])
    from atomic_sensor_simulation.kalman_filter import compute_B_from_d_vals
    Fs = [np.exp(np.array([[-atoms_correlation_const*dt, dt * CONSTANTS.g_a_COUPLING_CONST], [0, dt]])) for _ in range(num_iter)]
    us = [np.array([0, (amplitude/omega)*(np.cos(omega*(_))-np.cos(omega*(_-1)))]).T for _ in range(num_iter)]
    # Bs = [compute_B_from_d_vals(d_transition_matrix,
    #                             np.array([[create_multiplicable_const_func(1), create_multiplicable_const_func(0)], [create_multiplicable_const_func(0), create_multiplicable_const_func(1)]]),
    #                             _) for _ in range(num_iter)]
    Bs = [F.dot(np.eye(2)) for _ in range(num_iter)]

    from atomic_sensor_simulation.kalman_filter import initialize_kalman_filter_from_derrivatives

    kalman_filter = initialize_kalman_filter_from_derrivatives(np.array([spin_initial_val, quadrature_initial_val]))
    (mu, cov, _, _) = kalman_filter.batch_filter(zs, Fs=Fs, us=us, Bs=Bs)
    filtered_signal = mu[:, 1]

    # # plot results
    plt.plot(range(num_iter), qs, label='Signal')  # sensor readings
    # plt.plot(range(num_iter), cov[:,1], 'b')  # sensor readings
    plt.plot(range(num_iter), filtered_signal, label='Filtered signal')  # sensor readings
    plt.legend()
    # plt.ylim(0.4,0.6)
    plt.show()

def run_tests(*args):
    #:TODO implement this function
    pass


if __name__ == "__main__":

    main()
