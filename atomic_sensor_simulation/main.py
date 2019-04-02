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
    from numpy.random import randn

    class PosSensor(object):
        def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
            self.vel = vel
            self.noise_std = noise_std
            self.pos = [pos[0], pos[1]]

        def read(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]

            return [self.pos[0] + randn() * self.noise_std,
                    self.pos[1] + randn() * self.noise_std]

    import matplotlib.pyplot as plt
    import numpy as np

    pos, vel = (4, 3), (2, 1)
    sensor = PosSensor(pos, vel, noise_std=1)
    ps = np.array([sensor.read() for _ in range(50)])
    plt.plot(ps[:, 0], ps[:, 1])
    plt.show()

    from filterpy.kalman import KalmanFilter

    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.  # time step 1 second

    tracker.F = np.array([[1, dt, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]])

    from scipy.linalg import block_diag
    from filterpy.common import Q_discrete_white_noise

    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    tracker.Q = block_diag(q, q)
    print(tracker.Q)

    tracker.H = np.array([[1 / 0.3048, 0, 0, 0],
                          [0, 0, 1 / 0.3048, 0]])

    tracker.R = np.array([[5., 0],
                          [0, 5]])
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.

    from filterpy.stats import plot_covariance_ellipse

    R_std = 0.35
    Q_std = 0.04

    def tracker1():
        tracker = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # time step

        tracker.F = np.array([[1, dt, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, dt],
                              [0, 0, 0, 1]])
        tracker.u = 0.
        tracker.H = np.array([[1 / 0.3048, 0, 0, 0],
                              [0, 0, 1 / 0.3048, 0]])

        tracker.R = np.eye(2) * R_std ** 2
        q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std ** 2)
        tracker.Q = block_diag(q, q)
        tracker.x = np.array([[0, 0, 0, 0]]).T
        tracker.P = np.eye(4) * 500.
        return tracker

    # simulate robot movement
    N = 30
    sensor = PosSensor((0, 0), (2, .2), noise_std=R_std)

    zs = np.array([sensor.read() for _ in range(N)])

    # run filter
    robot_tracker = tracker1()
    mu, cov, _, _ = robot_tracker.batch_filter(zs)

    for x, P in zip(mu, cov):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[2, 0]],
                        [P[0, 2], P[2, 2]]])
        mean = (x[0, 0], x[2, 0])
        # plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)

    # plot results
    zs *= .3048  # convert to meters
    plt.plot(mu[:, 0], mu[:, 2])
    plt.scatter(zs[:, 0], zs[:, 1])
    plt.legend(loc=2)
    plt.xlim(0, 20)
    plt.show()

def run_wiener(*args):

    def linear_kalman():
        kalman_filter = KalmanFilter(dim_x=2, dim_z=1)
        kalman_filter.x = np.array([spin_initial_val, quadrature_initial_val])
        kalman_filter.F = np.array([[1, dt * CONSTANTS.g_a_COUPLING_CONST], [0, 1]])
        kalman_filter.H = np.array([[CONSTANTS.g_d_COUPLING_CONST * dt, 0]])
        kalman_filter.P *= CONSTANTS.SCALAR_STREGTH_y
        kalman_filter.R = np.array([[CONSTANTS.SCALAR_STREGTH_y]])
        from filterpy.common import Q_discrete_white_noise
        # kalman_filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
        kalman_filter.Q = block_diag(CONSTANTS.SCALAR_STREGTH_j, CONSTANTS.SCALAR_STRENGTH_q)
        return kalman_filter

    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-wiener command.')

    # initial conditions
    spin_initial_val = 0.0
    quadrature_initial_val = 0.5
    dt = 0.01
    num_iter = 2000

    state = State(spin=spin_initial_val,
                  quadrature=quadrature_initial_val,
                  noise_spin=GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STREGTH_j, dt=dt),
                  noise_quadrature=GaussianWhiteNoise(spin_initial_val, scalar_strength=CONSTANTS.SCALAR_STRENGTH_q, dt=dt),
                  dt=dt)
    sensor = AtomicSensor(state, scalar_strenght_y = CONSTANTS.SCALAR_STREGTH_y, dt=dt)

    zs, zs_no_noise, noise = zip(*np.array([(*sensor.read(), sensor.noise) for _ in range(num_iter)])) # read photocurrent values from the sensor
    # zs_minus_noise = np.subtract(zs, noise)
    # # plot (time, photocurrent) for values with noise and for values without noise
    # plt.plot(range(num_iter), zs, 'r', label='Noisy sensor detection')
    # # plt.plot(range(num_iter), noise, 'b', label='Noise')
    # plt.plot(range(num_iter), zs_minus_noise, 'g', label='Noisy sensor detection minus noise')
    # plt.plot(range(num_iter), zs_no_noise, 'k', label='Ideal data')
    # plt.legend()
    # plt.show()
    Fs = [np.array([[1., _ * CONSTANTS.g_a_COUPLING_CONST], [0, 1]]) for _ in range(num_iter)]
    kalman_filter = linear_kalman()
    (mu, cov, _, _) = kalman_filter.batch_filter(zs, Fs=Fs)
    (xs, Ps, Ks, _) = kalman_filter.rts_smoother(mu, cov, Fs=Fs)
    filtered_signal = mu[:, 1]

    # # plot results
    plt.plot(range(num_iter), zs_no_noise, 'k')  # sensor readings
    plt.plot(range(num_iter), np.ones(num_iter)*0.5, 'b')  # sensor readings
    plt.plot(range(num_iter), filtered_signal, 'r')  # sensor readings
    print(zs_no_noise)
    plt.show()

def run_tests(*args):
    #:TODO implement this function
    pass


if __name__ == "__main__":

    main()
