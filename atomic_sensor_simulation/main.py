#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import multiprocessing
import itertools
import time

from atomic_sensor_simulation.utilities import stringify_namespace, load_logging_config
from atomic_sensor_simulation.listener import listener_process
from atomic_sensor_simulation.run_atomic_sensor import run__atomic_sensor
from atomic_sensor_simulation.run_frequency_extractor import run__frequency_extractor
from atomic_sensor_simulation.run_tests import run_tests
from atomic_sensor_simulation.utilities import import_config_from_path, get_configs_from_config


def main():
    # setup a logger
    load_logging_config()
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of the simulation.')

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Atomic Sensor Simulation')
    # parser.add_argument('--working_dir', action='store', help='foo help')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    # create the parser for the "run-tests" command
    tests_parser = subparsers.add_parser('run-tests', help='Run unit tests')
    tests_parser.set_defaults(func=run_tests)

    # create the parser for the "run-simulation" command
    simulation_parser = subparsers.add_parser('run-atomic-sensor',
                                              help='Run atomic sensor simulation')
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
    simulation_parser.add_argument('--max_num_processes',
                                   action='store',
                                   help='Int specifying the maximum number of processes that should'
                                        'be spawned while running the simulation.',
                                   default=8)
    simulation_parser.set_defaults(func=run__atomic_sensor)

    # create the parser for the "run-frequency-extractor" command
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

    # parse some argument lists
    args = parser.parse_args()
    logger.info('Parsed input arguments %r' % stringify_namespace(args))

    # if running atomic sensor simulation spawn multiple threads, prepare a config for each thread
    if args.command == 'run-atomic-sensor':
        logger.info('Starting execution of run-atomic-sensor command.')
        config = import_config_from_path(args.config)
        configs = get_configs_from_config(config)

        logger.info('Preparing multiprocessing Queue.')
        args_list = itertools.repeat(args, len(configs))
        args_tuples = tuple(zip(configs, args_list))
        queue = multiprocessing.Queue(len(args_tuples))
        queue.put(list(args_tuples))
        # listener_process(queue)

        logger.info('Preparing multiprocessing Processes.')
        start_time = time.time()
        num_processes = args.max_num_processes if len(configs) >= args.max_num_processes else len(configs)
        workers = []
        for i in range(num_processes):
            worker = multiprocessing.Process(target=run__atomic_sensor, args=(queue,))
            workers.append(worker)
            worker.start()
        for w in workers:
            w.join()

        # pool = multiprocessing.Pool(processes=processes)
        # simulation_results = pool.starmap(run__atomic_sensor, args_tuples)
        # pool.close()
        # pool.join()
        logger.info('Simulation with %s processes spawned finished in %s' % (str(num_processes),
                                                                             str(time.time() - start_time)))
        # if any(simulation_results) != 0:
        #     logger.warning('Exit code other than 0 detected.')
        #     raise UserWarning('Not all simulation exit codes were 0.')
    else:
        args.func(args)

    logger.info('Ending execution of the application.')
    return 0


if __name__ == "__main__":
    main()
