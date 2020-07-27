#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np

from atomic_sensor_simulation.utilities import stringify_namespace, load_logging_config
from atomic_sensor_simulation.run_atomic_sensor import run__atomic_sensor
from atomic_sensor_simulation.run_frequency_extractor import run__frequency_extractor
from atomic_sensor_simulation.run_tests import run_tests


def main():
    # setup a logger
    load_logging_config()
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of the simulation.')

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Atomic Sensor Simulation')
    # parser.add_argument('--working_dir', action='store', help='foo help')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "run-tests" command
    tests_parser = subparsers.add_parser('run-tests', help='Run unit tests')
    tests_parser.set_defaults(func=run_tests)

    # create the parser for the "run-simulation" command
    simulation_parser = subparsers.add_parser('run-atomic-sensor', help='Run atomic sensor simulation')
    simulation_parser.add_argument('-o',
                                   '--output_path',
                                   action='store',
                                   help='A string representing path where the output should be saved.',
                                   default='./')
    simulation_parser.add_argument('--config',
                                   action='store',
                                   help='A string representing a module name of a config file. Config is a python file.',
                                   default='config')
    simulation_parser.add_argument('--lkf_num',
                                   action='store_true',
                                   help='Plot Linear kf',
                                   default=False)
    simulation_parser.add_argument('--lkf_expint',
                                   action='store_true',
                                   help='Plot Linear kf with solving for Phi using exp(integral Fdt) approx.',
                                   default=False)
    simulation_parser.add_argument('--lkf_exp',
                                   action='store_true',
                                   help='Plot Linear kf with solving for Phi using exp(Fdt) approx.',
                                   default=False)
    simulation_parser.add_argument('--ekf',
                                   action='store_true',
                                   help='Extended kf.',
                                   default=False)
    simulation_parser.add_argument('--ukf',
                                   action='store_true',
                                   help='Unscented kf.',
                                   default=False)
    simulation_parser.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    simulation_parser.add_argument('--save_data_file',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    simulation_parser.set_defaults(func=run__atomic_sensor)

    freq_extractor_parser = subparsers.add_parser('run-frequency-extractor', help='Run frequency extractor')
    freq_extractor_parser.add_argument('--config',
                                   action='store',
                                   help='A string representing a module name of a config file. Config is a python file.',
                                   default='config_freq')
    freq_extractor_parser.add_argument('--lkf_num',
                                   action='store_true',
                                   help='Plot Linear kf',
                                   default=False)
    freq_extractor_parser.add_argument('--ekf',
                                   action='store_true',
                                   help='Extended kf.',
                                   default=False)
    freq_extractor_parser.add_argument('--ukf',
                                   action='store_true',
                                   help='Unscented kf.',
                                   default=False)
    freq_extractor_parser.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    freq_extractor_parser.add_argument('--save_data_file',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    freq_extractor_parser.set_defaults(func=run__frequency_extractor)

    tests_parser = subparsers.add_parser('run-tests', help='')
    tests_parser.set_defaults(func=run_tests)

    # parse some argument lists
    args = parser.parse_args()
    logger.info('Parsed input arguments %r' % stringify_namespace(args))
    # for element in [150]:
    for wp in np.arange(1., 10., 1.).tolist():
        logger.info("Setting omega_p to %r" % wp)
        for gp in np.arange(5, 170, 20).tolist():
            logger.info("Setting g_p to %r" % gp)

            args.gp = gp
            args.wp = wp
            args.func(args)
    logger.info('Ending execution of the application.')
    return 0


if __name__ == "__main__":
    main()
