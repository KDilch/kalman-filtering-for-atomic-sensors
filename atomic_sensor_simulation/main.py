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
    # #:TODO implement this function (outdated due to the newest modifications of the code)
    # logger = logging.getLogger(__name__)
    # logger.info('Starting execution of run-simulation command.')
    # noise = GaussianWhiteNoise(initial_value=1, scalar_strength=1, dt=1)
    # noise.plot(is_show=False)
    # print(CONSTANTS.LARMOR_FREQ, CONSTANTS.T2_PARAM)
    # spin = Spin(larmor_freq=CONSTANTS.LARMOR_FREQ, t2_param=CONSTANTS.T2_PARAM, scalar_strength=1, dt=0.1)
    # quadrature = Quadrature(p=0,
    #                         q=1,
    #                         correlation_time=CONSTANTS.CORRELATION_CONSTANT_OU_PROCESS,
    #                         scalar_strength=1,
    #                         dt=CONSTANTS.SAMPLING_PERIOD) #initialize quadrature so that cos(t) is not there and sin(t) can be treated linearly
    # signal = Signal(CONSTANTS.LARMOR_FREQ, quadrature, dt=CONSTANTS.SAMPLING_PERIOD, coupling_const=CONSTANTS.CORRELATION_CONSTANT_OU_PROCESS)
    # state = State(spin, signal)
    # logger.info('Initialized state to %r'%state.__str__())
    # print(CONSTANTS.g_D_COUPLING_CONST, CONSTANTS.SCALAR_STREGTH_Y)
    # num_current = 0
    # import numpy as np
    # data = np.empty(1000)
    # while num_current < CONSTANTS.NUM_STEPS:
    #     state.step()
    #     data[num_current] = state.signal
    #     num_current += 1
    # print(data)
    # from atomic_sensor_simulation.utilities import plot_data
    # plot_data(np.arange(1000), data, is_show=True)
    # return

def run_wiener(*args):

    def linear_kalman():
        kalman_filter = KalmanFilter(dim_x=2, dim_z=1)
        kalman_filter.F = np.array([[1, dt * CONSTANTS.g_a_COUPLING_CONST], [0, 1]])
        kalman_filter.u = 0.
        kalman_filter.H = np.array([[CONSTANTS.g_d_COUPLING_CONST * dt, 0]])
        kalman_filter.R = np.array([[CONSTANTS.SCALAR_STREGTH_y]])
        kalman_filter.Q = block_diag(CONSTANTS.SCALAR_STREGTH_j, CONSTANTS.SCALAR_STRENGTH_q)
        kalman_filter.x = np.array([[spin_initial_val, quadradure_initial_val]]).T
        kalman_filter.P = np.eye(2) * CONSTANTS.SCALAR_STREGTH_y
        return kalman_filter

    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-wiener command.')

    # initial conditions
    spin_initial_val = 0.0
    quadradure_initial_val = 0.5
    dt = 0.1
    num_iter = 10000

    state = State(spin=spin_initial_val,
                  quadrature=quadradure_initial_val,
                  noise_spin=GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STREGTH_j, dt=dt),
                  noise_quadrature=GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STRENGTH_q, dt=dt),
                  dt=dt)
    sensor = AtomicSensor(state, scalar_strenght_y = CONSTANTS.SCALAR_STREGTH_y, dt=dt)

    zs = np.array([sensor.read() for _ in range(num_iter)])  # read photocurrent values from the sensor
    zs_no_noise = np.array([sensor.read_no_noise() for _ in range(num_iter)])  # noise free photocurrent values

    # plot (time, photocurrent) for values with noise and for values without noise
    plt.plot(range(num_iter), zs, 'r', label='Noisy sensor detection')
    plt.plot(range(num_iter), zs_no_noise, 'k', label='Ideal data')
    plt.legend()
    plt.show()

    # check if mean and variance of noise is OK (move it to tests in the future)
    noise = GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STREGTH_y, dt=dt)
    print(np.mean(noise.generate(10000)[1]))

    # TODO run filter
    # kalman_filter = linear_kalman()
    # mu, cov, _, _ = kalman_filter.batch_filter(zs)
    # def get_y(x, dt, y):
    #     y += CONSTANTS.g_d_COUPLING_CONST * x[0] * dt
    #     return y
    # ys = np.empty(num_iter)
    # y_current = 0.
    # count = 0
    # for _ in mu:
    #     y = get_y(_, dt, y_current)
    #     y_current = y
    #     ys[count] = y
    #     count += 1
    #
    # for x, P in zip(mu, cov):
    #     # covariance of x and q
    #     cov = np.array([[P[0, 0], P[1, 0]],
    #                     [P[0, 1], P[1, 1]]])
    #     mean = (x[0, 0], x[1, 0])
    #     # plot_covariance(mean, cov=cov, fc='g', std=CONSTANTS.SCALAR_STREGTH_y, alpha=0.0)
    #
    # # # plot results
    # from atomic_sensor_simulation.utilities import plot_data
    # plt.plot(range(num_iter), zs, 'r', range(num_iter), ys, 'b', range(num_iter), zs_no_noise, 'k')  # sensor readings
    # plt.show()

def run_tests(*args):
    #:TODO implement this function
    pass


if __name__ == "__main__":

    main()
