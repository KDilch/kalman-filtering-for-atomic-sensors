#!/usr/bin/env python
# -*- coding: utf-8 -*-
from atomic_sensor_simulation.utilities import stringify_namespace, load_logging_config
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.state import Spin, State, Signal, Quadrature
from atomic_sensor_simulation import CONSTANTS
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

    # parse some argument lists
    args = parser.parse_args()
    logger.info('Parsed input arguments %r' % stringify_namespace(args))
    args.func(args)
    logger.info('Ending execution of the application.')
    return 0


def run_simulation(*args):
    #:TODO implement this function
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-simulation command.')
    noise = GaussianWhiteNoise(initial_value=1, scalar_strength=1, dt=1)
    noise.plot(is_show=False)
    print(CONSTANTS.LARMOR_FREQ, CONSTANTS.T2_PARAM)
    spin = Spin(larmor_freq=CONSTANTS.LARMOR_FREQ, t2_param=CONSTANTS.T2_PARAM, scalar_strength=1, dt=0.1)
    quadrature = Quadrature(p=0,
                            q=1,
                            correlation_time=CONSTANTS.CORRELATION_CONSTANT_OU_PROCESS,
                            scalar_strength=1,
                            dt=CONSTANTS.SAMPLING_PERIOD) #initialize quadrature so that cos(t) is not there and sin(t) can be treated linearly
    signal = Signal(CONSTANTS.LARMOR_FREQ, quadrature, dt=CONSTANTS.SAMPLING_PERIOD, coupling_const=CONSTANTS.CORRELATION_CONSTANT_OU_PROCESS)
    state = State(spin, signal)
    logger.info('Initialized state to %r'%state.__str__())
    print(CONSTANTS.g_D_COUPLING_CONST, CONSTANTS.SCALAR_STREGTH_Y)
    num_current = 0
    import numpy as np
    data = np.empty(1000)
    while num_current < CONSTANTS.NUM_STEPS:
        state.step()
        data[num_current] = state.signal
        num_current += 1
    print(data)
    from atomic_sensor_simulation.utilities import plot_data
    plot_data(np.arange(1000), data, is_show=True)
    return


def run_tests(*args):
    #:TODO implement this function
    pass


if __name__ == "__main__":

    main()
