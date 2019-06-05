#!/usr/bin/env python
# -*- coding: utf-8 -*-
from atomic_sensor_simulation.utilities import stringify_namespace, load_logging_config
from atomic_sensor_simulation.noise import GaussianWhiteNoise

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
    simulation_parser.set_defaults(func=run__atomic_sensor)

    tests_parser = subparsers.add_parser('run-atomic-sensor', help='')
    tests_parser.set_defaults(func=run__atomic_sensor)

    position_speed_parser = subparsers.add_parser('run-position-speed', help='')
    position_speed_parser.set_defaults(func=run_position_speed)

    # parse some argument lists
    args = parser.parse_args()
    logger.info('Parsed input arguments %r' % stringify_namespace(args))
    args.func(args)
    logger.info('Ending execution of the application.')
    return 0


def run__atomic_sensor(*args):
    from atomic_sensor_simulation.state.atomic_sensor import AtomicSensorState
    from atomic_sensor_simulation.sensor.atomic_sensor import AtomicSensor
    from atomic_sensor_simulation.model.atomic_sensor_model import AtomicSensorModel

    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-atomic-sensor command.')

    # SIMULATION=====================================================
    # simulation parameters
    num_iter = 300
    dt = 1.
    atoms_correlation_const = 0.1
    logger.info('Setting simulation parameters to num_iter = %r, delta_t = %r, atoms_correlation_const=%r.' %
                (str(num_iter),
                 str(dt),
                 str(atoms_correlation_const)
                 )
                )
    g_a_COUPLING_CONST = 0.2
    SCALAR_STREGTH_y = 0.01
    SCALAR_STREGTH_j = 0.01
    SCALAR_STRENGTH_q = 0.01

    time_arr = np.arange(0, num_iter, dt)

    #consts for control function -> amplitude*sin(omega*t)
    omega = 0.1
    amplitude = 1.

    #initial conditions
    spin_initial_val = 0.0
    quadrature_initial_val = 0.1

    logger.info('Setting initial conditions to spin = %r, quadrature = %r' %
                (str(spin_initial_val),
                 str(quadrature_initial_val)
                 )
                )

    state = AtomicSensorState(initial_vec=np.array([spin_initial_val, quadrature_initial_val]),
                              noise_vec=np.array([GaussianWhiteNoise(spin_initial_val,
                                                                     scalar_strength=SCALAR_STREGTH_j,
                                                                     dt=dt),
                                                  GaussianWhiteNoise(spin_initial_val,
                                                                     scalar_strength=SCALAR_STRENGTH_q,
                                                                     dt=dt)]),
                              initial_time=0,
                              atoms_wiener_const=atoms_correlation_const,
                              g_a_coupling_const=g_a_COUPLING_CONST,
                              control_amplitude=amplitude,
                              control_freq=omega)

    sensor = AtomicSensor(state,
                          scalar_strenght_y=SCALAR_STREGTH_y,
                          dt=dt)

    zs = np.array([np.array((sensor.read(_))) for _ in range(num_iter)])  # noisy measurement

    # KALMAN FILTER=====================================================
    model = AtomicSensorModel(F=np.array(state.F_evolution_matrix(0), dtype='float64'),
                              initial_state=np.array([0.1, 0.1]),
                              dim_z=1)

    (mu, cov, _, _) = model.filterpy.batch_filter(zs)
    filtered_data = mu[:, 1]

    # plot data (noisy, exact and filtered)
    plt.plot(time_arr, filtered_data, label='Filtered data')
    plt.plot(range(num_iter), sensor.quadrature_no_noise_full_history, label='Exact data')
    plt.plot(time_arr, sensor.quadrature_full_history, label='Noisy data')
    plt.legend()
    plt.show()

    # plot covariances
    plt.plot(range(num_iter), cov[:, 1, 1], 'b')  # covariances P_q
    plt.show()

    # Stable solution

def run_position_speed(*args):
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run_position_speed command.')
    dt = 1.
    num_iter = 20

    from atomic_sensor_simulation.state.pos_vel_sensor import PosVelSensorState

    state = PosVelSensorState(initial_vec=np.array([0., 0., 2., 1.]),
                              noise_vec=np.array([GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt),
                                                  GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt),
                                                  GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt),
                                                  GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt)
                                                  ]),
                              initial_time=0)
    from atomic_sensor_simulation.sensor import pos_sensor

    sensor = pos_sensor.PosSensor(state,
                                  scalar_strenght_y=1.,
                                  dt=dt)
    time_arr = np.arange(0, num_iter, dt)

    zs = np.array([(np.array([sensor.read(_)]).T) for _ in time_arr])  # read from the sensor

    # KALMAN FILTER
    # Fs = [state.F_evolution_matrix(_) for _ in time_arr]
    # us = [state.u_control_vec(time) for time in time_arr]
    # Bs = [state.B_control_evolution_matrix(time) for time in time_arr]

    # waveform = np.empty_like(us)
    # for element in range(len(us)):
    #     waveform[element] = Bs[element].dot(us[element])[1]

    kalman_filter = initialize_kalman_filter_from_derrivatives(np.array([0., 0., 0., 0.]).T,
                                                               dt=dt,
                                                               initial_F=state.F_evolution_matrix(0))
    (mu, cov, _, _) = kalman_filter.batch_filter(zs)
    filtered_signal = mu[:, 0]
    zs *= .3048
    # # plot results
    # plt.plot(range(num_iter), sensor.quadrature_full_history, label='Signal')  # sensor readings
    # plt.plot(range(num_iter), cov[:,1], 'b')  # covariances
    plt.plot(mu[:, 0], mu[:, 2], label='Filtered signal')  # filtered signal
    # plt.plot(range(num_iter), waveform, label=' quadrature_no_noise')  # waveform
    plt.plot(zs[:, 0], zs[:, 1], label=' quadrature_history')  # waveform
    plt.legend()
    plt.show()


def run_tests(*args):
    #:TODO implement
    pass


if __name__ == "__main__":

    main()
