#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logger import Logger
from utilities import stringify_namespace
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
    tests_parser = subparsers.add_parser('run_tests', help='Run unit tests')
    tests_parser.set_defaults(func=run_tests)

    # create the parser for the "run-simulation" command
    simulation_parser = subparsers.add_parser('run_simulation', help='Run atomic sensor simulation')
    simulation_parser.add_argument('--output_path',
                                   action='store',
                                   help='A string representing path where the output should be saved.',
                                   default='./')
    simulation_parser.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    simulation_parser.add_argument('--save_data_file',
                                   action='store_true',
                                   help='Bool specifying if you want to data file',
                                   default=False)
    simulation_parser.set_defaults(func=run_simulation)

    # parse some argument lists
    args = parser.parse_args()
    args.func(args)

    print(stringify_namespace(args))
    logs.logger.info('Parsed input arguments %r' % stringify_namespace(args))

    return 0


def run_simulation(*args):
    pass


def run_tests(*args):
    pass


if __name__ == "__main__":

    main()
