#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logger import Logger
from utilities import stringify_namespace
from noisy_measurement import NoisyDataGenerator
from atomic_sensor import Signal
import argparse


def main():
    # setup a logger
    logs = Logger('atomic_sensor_simulation', log_file_path='logs/atomic_sensor_simulation.log')
    logs.logger.info('Starting execution of Atomic Sensor Simulation.')

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
    args.func(args)
    logs.logger.info('Parsed input arguments %r' % stringify_namespace(args))

    return 0


def run_simulation(*args):
    #:TODO implement this function
    logs = Logger('run-simulation-logger', log_file_path='logs/atomic_sensor_simulation.log')
    logs.logger.info('Starting execution of run-simulation command.')
    clean_signal = Signal(larmor_freq=1, quadrature=)
    noise = NoisyDataGenerator(logs=logs, signal=clean_signal, time_step=1)
    pass


def run_tests(*args):
    #:TODO implement this function
    pass


if __name__ == "__main__":

    main()
