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
    simulation_parser.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    simulation_parser.add_argument('--save_data_file',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    simulation_parser.set_defaults(func=run__atomic_sensor)

    tests_parser = subparsers.add_parser('run-tests', help='')
    tests_parser.set_defaults(func=run_tests)

    # parse some argument lists
    args = parser.parse_args()
    logger.info('Parsed input arguments %r' % stringify_namespace(args))
    args.func(args)
    logger.info('Ending execution of the application.')
    return 0


def run__atomic_sensor(*args):
    from atomic_sensor_simulation.utilities import import_config_from_path
    from atomic_sensor_simulation.state.atomic_state import AtomicSensorState
    from atomic_sensor_simulation.sensor.atomic_sensor import AtomicSensor
    from atomic_sensor_simulation.filter_model.linear_kf import Linear_KF
    from atomic_sensor_simulation.filter_model.unscented_kf import Unscented_KF
    from atomic_sensor_simulation.filter_model.extented_kf import Extended_KF
    from atomic_sensor_simulation.utilities import calculate_error, compute_squred_error_from_covariance, \
        eval_matrix_of_functions
    from atomic_sensor_simulation.history_manager import Filter_History_Manager
    from atomic_sensor_simulation.atomic_sensor_steady_state import compute_steady_state_solution_for_atomic_sensor

    from scipy.linalg import expm

    # Logger for storing errors and logs in seprate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-atomic-sensor command.')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)

    logger.info(
        'Setting physical parameters to larmour_freq = %r, spin_correlation_const = %r, light_correlation_const=%r.' %
        (str(config.physical_parameters['larmour_freq']),
         str(config.physical_parameters['spin_correlation_const']),
         str(config.physical_parameters['light_correlation_const'])
         )
        )

    logger.info('Setting simulation parameters to delta_t_sensor = %r, number_periods=%r.' %
                (str(config.simulation['dt_sensor']),
                 str(config.simulation['number_periods'])
                 )
                )

    logger.info('Setting filter parameters to delta_t_filter = %r.' %
                (str(config.filter['dt_filter'])
                 )
                )

    logger.info('Setting initial state vec to  [%r, %r, %r, %r].' %
                (str(config.simulation['spin_y_initial_val']),
                 str(config.simulation['spin_z_initial_val']),
                 str(config.simulation['q_initial_val']),
                 str(config.simulation['p_initial_val'])
                 )
                )

    num_iter_sensor = (2 * np.pi * config.simulation['number_periods'] /
                       config.physical_parameters['larmour_freq']) / config.simulation['dt_sensor']
    num_iter_filter = np.int(np.floor_divide(num_iter_sensor * config.simulation['dt_sensor'],
                                             config.filter['dt_filter']))

    every_nth_z = np.int(np.floor_divide(num_iter_sensor, num_iter_filter))

    Q = np.array([[config.noise_and_measurement['QJy'], 0., 0., 0.],
                  [0., config.noise_and_measurement['QJz'], 0., 0.],
                  [0., 0., config.noise_and_measurement['Qq'], 0.],
                  [0., 0., 0., config.noise_and_measurement['Qp']]])
    H = np.array([[0., config.noise_and_measurement['gD'], 0., 0.]])
    R = np.array([[config.noise_and_measurement['QD']]])

    logger.info('Setting Q, H and R to Q = %r, H = %r, R = %r' %
                (str(Q),
                 str(H),
                 str(R)
                 )
                )

    time_arr = np.arange(0, num_iter_sensor * config.simulation['dt_sensor'], config.simulation['dt_sensor'])
    time_arr_filter = np.arange(0, num_iter_filter * config.filter['dt_filter'], config.filter['dt_filter'])

    # SIMULATING DYNAMICS=====================================================

    state = AtomicSensorState(initial_vec=np.array([config.simulation['spin_y_initial_val'],
                                                    config.simulation['spin_z_initial_val'],
                                                    config.simulation['q_initial_val'],
                                                    config.simulation['p_initial_val']]),
                              noise_vec=GaussianWhiteNoise(mean=[0., 0., 0., 0.],
                                                           cov=Q,
                                                           dt=config.simulation['dt_sensor']),
                              initial_time=0,
                              dt=config.simulation['dt_sensor'],
                              light_correlation_const=config.physical_parameters['light_correlation_const'],
                              spin_correlation_const=config.physical_parameters['spin_correlation_const'],
                              larmour_freq=config.physical_parameters['larmour_freq'],
                              coupling_amplitude=config.coupling['g_p'],
                              coupling_freq=config.coupling['omega_p'],
                              coupling_phase_shift=config.coupling['phase_shift'])

    sensor = AtomicSensor(state,
                          sensor_noise=GaussianWhiteNoise(mean=0.,
                                                          cov=R / config.simulation['dt_sensor'],
                                                          dt=config.simulation['dt_sensor']),
                          H=H,
                          dt=config.simulation['dt_sensor'])

    zs = np.array([np.array((sensor.read(_))) for _ in time_arr])  # noisy measurement
    zs_filter_freq = zs[::every_nth_z]
    x_filter_freq = sensor.state_vec_full_history[::every_nth_z]

    # KALMAN FILTER====================================================
    # Linear Kalman Filter
    linear_kf_model = Linear_KF(F=state.F_transition_matrix,
                                Q=Q,
                                H=H,
                                R=R / config.filter['dt_filter'],
                                Gamma=state.Gamma_control_evolution_matrix,
                                u=state.u_control_vec,
                                z0=[zs[0]],
                                dt=config.filter['dt_filter'],
                                x0=np.array([config.filter['spin_y_initial_val'],
                                             config.filter['spin_z_initial_val'],
                                             config.filter['q_initial_val'],
                                             config.filter['p_initial_val']]),
                                P0=config.filter['P0'],
                                light_correlation_const=config.physical_parameters['light_correlation_const'],
                                spin_correlation_const=config.physical_parameters['spin_correlation_const'],
                                larmour_freq=config.physical_parameters['larmour_freq'],
                                coupling_amplitude=config.coupling['g_p'],
                                coupling_freq=config.coupling['omega_p'],
                                coupling_phase_shift=config.coupling['phase_shift']
                                )

    def compute_fx_at_time_t(t):
        F_t = eval_matrix_of_functions(state._F_transition_matrix, t)

        def fx(x, dt):
            return x + F_t.dot(x) * dt

        return fx

    def hx(x):
        return H.dot(x)

    def HJacobianat(x):
        return H

    unscented_kf_model = Unscented_KF(fx=compute_fx_at_time_t(0),
                                      Q=linear_kf_model.Q_delta,
                                      hx=hx,
                                      R=R / config.filter['dt_filter'],
                                      Gamma=state.Gamma_control_evolution_matrix,
                                      u=state.u_control_vec,
                                      z0=[zs[0]],
                                      dt=config.filter['dt_filter'],
                                      x0=linear_kf_model.x0,
                                      P0=linear_kf_model.P0)

    extended_kf_model = Extended_KF(F=state.F_transition_matrix,
                                    Q=linear_kf_model.Q_delta,
                                    hx=hx,
                                    R=R / config.filter['dt_filter'],
                                    Gamma=state.Gamma_control_evolution_matrix,
                                    u=state.u_control_vec,
                                    z0=[zs[0]],
                                    dt=config.filter['dt_filter'],
                                    x0=linear_kf_model.x0,
                                    P0=linear_kf_model.P0,
                                    num_terms=3
                                    )

    # RUN FILTERPY KALMAN FILTER
    logger.info("Initializing linear_kf_filterpy Kalman Filter")
    linear_kf_filterpy = linear_kf_model.initialize_filterpy()
    linear_kf_history_manager = Filter_History_Manager(linear_kf_filterpy, num_iter_filter)

    logger.info("Initializing unscented_kf_filterpy Unscented Filter")
    unscented_kf_filterpy = unscented_kf_model.initialize_filterpy()
    unscented_kf_history_manager = Filter_History_Manager(unscented_kf_filterpy, num_iter_filter)

    logger.info("Initializing extended_kf_filterpy Unscented Filter")
    extended_kf_filterpy = extended_kf_model.initialize_filterpy(
        light_correlation_const=config.physical_parameters['light_correlation_const'],
        spin_correlation_const=config.physical_parameters['spin_correlation_const'],
        larmour_freq=config.physical_parameters['larmour_freq'],
        coupling_amplitude=config.coupling['g_p'],
        coupling_freq=config.coupling['omega_p'],
        coupling_phase_shift=config.coupling['phase_shift'])
    extended_kf_history_manager = Filter_History_Manager(extended_kf_filterpy, num_iter_filter)

    error_jy = np.zeros(num_iter_filter)
    error_jz = np.zeros(num_iter_filter)
    error_q = np.zeros(num_iter_filter)
    error_p = np.zeros(num_iter_filter)

    for index, time in enumerate(time_arr_filter):
        z = zs_filter_freq[index]
        # linear_kf_filterpy.predict(F=linear_kf_model.compute_Phi_delta_exp_Fdt_approx(from_time=time))
        linear_kf_filterpy.predict()
        linear_kf_filterpy.F = linear_kf_model.compute_Phi_delta_solve_ode_numerically(from_time=time,
                                                                                       Phi_0=linear_kf_filterpy.F)
        logger.info('Setting Phi to [%r]' % str(linear_kf_filterpy.F))
        linear_kf_filterpy.update(z)
        linear_kf_history_manager.add_entry(index)

        unscented_kf_filterpy.predict(fx=compute_fx_at_time_t(time))
        unscented_kf_filterpy.update(z)
        unscented_kf_history_manager.add_entry(index)

        extended_kf_filterpy.predict()
        extended_kf_filterpy.update(z, HJacobianat, hx)
        extended_kf_history_manager.add_entry(index)

        error_jy[index] = calculate_error(config.W['W_jy'], x=x_filter_freq[index], x_est=linear_kf_filterpy.x)
        error_jz[index] = calculate_error(config.W['W_jz'], x=x_filter_freq[index], x_est=linear_kf_filterpy.x)
        error_q[index] = calculate_error(config.W['W_q'], x=x_filter_freq[index], x_est=linear_kf_filterpy.x)
        error_p[index] = calculate_error(config.W['W_p'], x=x_filter_freq[index], x_est=linear_kf_filterpy.x)

    # FIND STEADY STATE SOLUTION
    steady_priors_jy = []
    steady_posts_jy = []
    steady_priors_jz = []
    steady_posts_jz = []
    steady_priors_p = []
    steady_posts_p = []
    steady_priors_q = []
    steady_posts_q = []
    for time_filter in time_arr_filter:
        steady_prior, steady_post = compute_steady_state_solution_for_atomic_sensor(t=time_filter,
                                                                                    F=eval_matrix_of_functions(
                                                                                        state.F_transition_matrix,
                                                                                        time_filter),
                                                                                    model=linear_kf_model)
        logger.info("Steady state solution: predict_cov=%r,\n update_cov=%r" % (steady_prior, steady_post))
        steady_priors_jy.append(steady_prior[0][0])
        steady_posts_jy.append(steady_post[0][0])
        steady_priors_jz.append(steady_prior[1][1])
        steady_posts_jz.append(steady_post[1][1])
        steady_priors_q.append(steady_prior[2][2])
        steady_posts_q.append(steady_post[2][2])
        steady_priors_p.append(steady_prior[3][3])
        steady_posts_p.append(steady_post[3][3])

    # PLOTS=========================================================
    # Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_q_full_history, q_p_full_history = zip(*sensor.state_vec_full_history)

    # plot atoms jy
    logger.info("Plotting data jy")
    plt.title("Atoms jy")
    plt.plot(time_arr_filter, linear_kf_history_manager.jys, label='Linear kf')
    plt.plot(time_arr_filter, unscented_kf_history_manager.jys, label='Unscented kf')
    plt.plot(time_arr_filter, extended_kf_history_manager.jys, label='Extended kf')
    plt.plot(time_arr, j_y_full_history, label='Exact data')
    plt.legend()
    plt.show()

    # plot error for atoms jy
    logger.info("Plotting error jy")
    plt.title("Squared error jy")
    # plt.plot(time_arr_filter, error_jy, label='Squared error jy')
    # plt.plot(time_arr_filter, linear_kf_error_jy_prior, label='Prior linear kf')
    plt.plot(time_arr_filter, linear_kf_history_manager.jys_err_post, label='Post linear kf')
    # plt.plot(time_arr_filter, unscented_kf_error_jy_prior, label='Prior unscented kf')
    plt.plot(time_arr_filter, unscented_kf_history_manager.jys_err_post, label='Post unscented kf')
    # plt.plot(time_arr_filter, extended_kf_error_jy_prior, label='Prior extended kf')
    plt.plot(time_arr_filter, extended_kf_history_manager.jys_err_post, label='Post extended kf')
    # plt.plot(time_arr_filter, steady_priors_jy, color='r', label='steady_prior')
    plt.plot(time_arr_filter, steady_posts_jy, color='b', label="steady_post")
    # plt.axhline(y=steady_post[0][0], color='r', linestyle='-', label='steady_post')
    # plt.axhline(y=steady_prior[0][0], color='b', linestyle='-', label='steady_prior')
    plt.legend()
    plt.show()

    # plot atoms jz
    logger.info("Plotting data jz")
    plt.title("Atoms jz")
    plt.plot(time_arr_filter, linear_kf_history_manager.jzs, label='Linear kf')
    # plt.plot(time_arr_filter, unscented_kf_atoms_jz, label='Unscented kf')
    # plt.plot(time_arr_filter, extended_kf_atoms_jz, label='Extended kf')
    plt.plot(time_arr, j_z_full_history, label='Exact data')
    plt.legend()
    plt.show()

    # plot error for atoms jz
    logger.info("Plotting error jz")
    plt.title("Squared error jz")
    # plt.plot(time_arr_filter, error_jz, label='Squared error jz')
    # plt.plot(time_arr_filter, linear_kf_error_jz_prior, label='Prior linear kf')
    plt.plot(time_arr_filter, linear_kf_history_manager.jzs_err_post, label='Post linear kf')
    # plt.plot(time_arr_filter, unscented_kf_error_jz_prior, label='Prior unscented kf')
    # plt.plot(time_arr_filter, unscented_kf_error_jz_post, label='Post unscented kf')
    # plt.plot(time_arr_filter, extended_kf_error_jz_prior, label='Prior extended kf')
    # plt.plot(time_arr_filter, extended_kf_error_jz_post, label='Post extended kf')
    # plt.plot(time_arr_filter, steady_priors_jz, color='r', label='steady_prior')
    plt.plot(time_arr_filter, steady_posts_jz, color='b', label="steady_posts")
    plt.legend()
    plt.show()

    # plot light q (noisy, exact and filtered)
    logger.info("Plotting data")
    plt.title("Light q")
    plt.plot(time_arr_filter, linear_kf_history_manager.qs, label='Linear kf')
    # plt.plot(time_arr_filter, unscented_kf_light_q, label='Unscented kf')
    # plt.plot(time_arr_filter, extended_kf_light_q, label='Extended kf')
    plt.plot(time_arr, q_q_full_history, label='Exact data')
    plt.legend()
    plt.show()

    # plot error q
    logger.info("Plotting error q")
    plt.title("Squared error q")
    # plt.plot(time_arr_filter, linear_kf_error_q_prior, label='Prior linear kf')
    plt.plot(time_arr_filter, linear_kf_history_manager.qs_err_post, label='Post linear kf')
    # plt.plot(time_arr_filter, unscented_kf_error_q_prior, label='Prior unscented kf')
    # plt.plot(time_arr_filter, unscented_kf_error_q_post, label='Post unscented kf')
    # plt.plot(time_arr_filter, extended_kf_error_q_prior, label='Prior extended kf')
    # plt.plot(time_arr_filter, extended_kf_error_q_post, label='Post extended kf')
    # plt.plot(time_arr_filter, error_q, label='Squared error q')
    # plt.plot(time_arr_filter, steady_priors_q, color='r', label='steady_prior')
    plt.plot(time_arr_filter, steady_posts_q, color='b', label="steady_posts")
    plt.legend()
    plt.show()

    # plot light p (noisy, exact and filtered)
    logger.info("Plotting data")
    plt.title("Light p")
    plt.plot(time_arr_filter, linear_kf_history_manager.ps, label='Linear kf')
    # plt.plot(time_arr_filter, unscented_kf_light_p, label='Unscented kf')
    # plt.plot(time_arr_filter, extended_kf_light_p, label='Extended kf')
    plt.plot(time_arr, q_p_full_history, label='Exact data')
    plt.legend()
    plt.show()

    # plot error p
    logger.info("Plotting error p")
    plt.title("Squared error p")
    # plt.plot(time_arr_filter, linear_kf_error_p_prior, label='Prior linear kf')
    plt.plot(time_arr_filter, linear_kf_history_manager.ps_err_post, label='Post linear kf')
    # plt.plot(time_arr_filter, steady_priors_p, color='r', label='steady_prior')
    plt.plot(time_arr_filter, steady_posts_p, color='b', label="steady_posts")
    # plt.plot(time_arr_filter, unscented_kf_error_p_prior, label='Prior unscented kf')
    # plt.plot(time_arr_filter, unscented_kf_error_p_post, label='Prior unscented kf')
    # plt.plot(time_arr_filter, extended_kf_error_p_prior, label='Prior extended kf')
    # plt.plot(time_arr_filter, extended_kf_error_p_post, label='Prior extended kf')
    # plt.plot(time_arr_filter, error_p, label='Squared error p')
    plt.legend()
    plt.show()

    # # plot zs
    # plt.plot(time_arr, sensor.z_no_noise_arr, label='Exact sensor data')
    # plt.plot(time_arr, zs, label='Noisy sensor readings')
    # plt.legend()
    # plt.show()


def run_tests(*args):
    from atomic_sensor_simulation.tests import test_main
    test_main.main()
    return 0


if __name__ == "__main__":
    main()
