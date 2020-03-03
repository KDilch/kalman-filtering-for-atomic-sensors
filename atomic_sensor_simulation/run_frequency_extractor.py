#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np

from atomic_sensor_simulation.utilities import import_config_from_path
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.state.atomic_state import AtomicSensorState
from atomic_sensor_simulation.sensor.atomic_sensor import AtomicSensor
from atomic_sensor_simulation.filter_model.linear_kf import Linear_KF
from atomic_sensor_simulation.filter_model.unscented_kf import Unscented_KF
from atomic_sensor_simulation.filter_model.extented_kf import Extended_KF
from atomic_sensor_simulation.utilities import calculate_error, eval_matrix_of_functions, plot_data, \
    generate_data_arr_for_plotting
from atomic_sensor_simulation.history_manager import Filter_History_Manager, SteadyStateHistoryManager
from atomic_sensor_simulation.atomic_sensor_steady_state import compute_steady_state_solution_for_atomic_sensor


def run__frequency_extractor(*args):

    # Logger for storing errors and logs in seprate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-frequency-extractor command.')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)

    logger.info('Setting simulation parameters to delta_t_sensor = %r, number_periods=%r.' %
                (str(config.simulation['dt_sensor']),
                 str(config.simulation['number_periods'])
                 )
                )

    logger.info('Setting filter parameters to delta_t_filter = %r.' %
                (str(config.filter['dt_filter'])
                 )
                )

    logger.info('Setting initial state vec to  [%r, %r, %r].' %
                (str(config.simulation['x1']),
                 str(config.simulation['x2']),
                 str(config.simulation['x3'])
                 )
                )

    num_iter_sensor = (2 * np.pi * config.simulation['number_periods'] /
                       config.physical_parameters['larmour_freq']) / config.simulation['dt_sensor']
    num_iter_filter = np.int(np.floor_divide(num_iter_sensor * config.simulation['dt_sensor'],
                                             config.filter['dt_filter']))

    every_nth_z = np.int(np.floor_divide(num_iter_sensor, num_iter_filter))

    Q = np.array([[config.noise_and_measurement['Qx1'], 0., 0., 0.],
                  [0., config.noise_and_measurement['Qx2'], 0., 0.],
                  [0., 0., config.noise_and_measurement['Qx3'], 0.]])
    H = np.array([[0., 0., 1.]])
    R = np.array([[config.noise_and_measurement['r']]])

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
                              time_arr=time_arr,
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
                                    num_terms=3,
                                    time_arr=time_arr_filter
                                    )

    # RUN FILTERPY KALMAN FILTER
    logger.info("Initializing linear_kf_filterpy Kalman Filter")

    lkf_exp_approx = linear_kf_model.initialize_filterpy()
    lkf_exp_approx_history_manager = Filter_History_Manager(lkf_exp_approx, num_iter_filter)

    lkf_expint_approx = linear_kf_model.initialize_filterpy()
    lkf_expint_approx_history_manager = Filter_History_Manager(lkf_expint_approx, num_iter_filter)

    lkf_num = linear_kf_model.initialize_filterpy()
    lkf_num_history_manager = Filter_History_Manager(lkf_num, num_iter_filter)

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
        lkf_num.predict()
        lkf_num.F = linear_kf_model.compute_Phi_delta_solve_ode_numerically(from_time=time)
        lkf_num.Q = linear_kf_model.compute_Q_delta_sympy(from_time=time,
                                                          Phi_0=lkf_num.F,
                                                          num_terms=30)
        lkf_expint_approx.predict()
        lkf_expint_approx.F = linear_kf_model.compute_Phi_delta_solve_ode_numerically(from_time=time)
        lkf_exp_approx.predict()
        lkf_exp_approx.F = linear_kf_model.compute_Phi_delta_solve_ode_numerically(from_time=time)
        logger.debug('Setting Phi to [%r]' % str(lkf_num.F))
        lkf_num.update(z)
        lkf_expint_approx.update(z)
        lkf_exp_approx.update(z)

        lkf_num_history_manager.add_entry(index)
        lkf_expint_approx_history_manager.add_entry(index)
        lkf_exp_approx_history_manager.add_entry(index)

        unscented_kf_filterpy.predict(fx=compute_fx_at_time_t(time))
        unscented_kf_filterpy.update(z)
        unscented_kf_history_manager.add_entry(index)

        extended_kf_filterpy.predict()
        extended_kf_filterpy.update(z, HJacobianat, hx)
        extended_kf_history_manager.add_entry(index)

        error_jy[index] = calculate_error(config.W['W_jy'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_jz[index] = calculate_error(config.W['W_jz'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_q[index] = calculate_error(config.W['W_q'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_p[index] = calculate_error(config.W['W_p'], x=x_filter_freq[index], x_est=lkf_num.x)

    # FIND STEADY STATE SOLUTION
    steady_state_history_manager = SteadyStateHistoryManager(num_iter_filter)
    for index, time_filter in enumerate(time_arr_filter):
        steady_prior, steady_post = compute_steady_state_solution_for_atomic_sensor(t=time_filter,
                                                                                    F=eval_matrix_of_functions(
                                                                                        state.F_transition_matrix,
                                                                                        time_filter),
                                                                                    model=linear_kf_model)
        logger.debug("Steady state solution: predict_cov=%r,\n update_cov=%r" % (steady_prior, steady_post))
        steady_state_history_manager.add_entry(steady_prior, steady_post, index)


    # PLOTS=========================================================
    # Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_q_full_history, q_p_full_history = zip(*sensor.state_vec_full_history)

    labels = ['Linear kf num', 'Linear kf expint', 'Linear kf exp', 'Extended kf',  'Unscented kf', 'Exact data']
    labels_err = ['Linear kf num err', 'Linear kf expint err', 'Linear kf exp err', 'Extended kf err',  'Unscented kf err', 'Steady state']

    # plot atoms jy
    if np.any([args[0].lkf_num, args[0].lkf_exp, args[0].lkf_expint, args[0].ekf, args[0].ukf]):
        logger.info("Plotting data jy")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                 time_arr_filter,
                                                 time_arr_filter,
                                                 time_arr_filter,
                                                 time_arr_filter,
                                                 time_arr]),
                                       np.array([lkf_num_history_manager.jys,
                                                 lkf_expint_approx_history_manager.jys,
                                                 lkf_exp_approx_history_manager.jys,
                                                 extended_kf_history_manager.jys,
                                                 unscented_kf_history_manager.jys,
                                                 j_y_full_history]),
                                       labels=labels,
                                       bools=[args[0].lkf_num,
                                              args[0].lkf_exp,
                                              args[0].lkf_expint,
                                              args[0].ekf,
                                              args[0].ukf,
                                              True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Atoms jy", is_show=True, is_legend=True)

        # plot error for atoms jy
        logger.info("Plotting error jy")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter]),
                                                                    np.array([lkf_num_history_manager.jys_err_post,
                                                                              lkf_expint_approx_history_manager.jys_err_post,
                                                                              lkf_exp_approx_history_manager.jys_err_post,
                                                                              extended_kf_history_manager.jys_err_post,
                                                                              unscented_kf_history_manager.jys_err_post,
                                                                              steady_state_history_manager.steady_posts_jy]),
                                                                    labels=labels_err,
                                                                    bools=[args[0].lkf_num,
                                                                           args[0].lkf_exp,
                                                                           args[0].lkf_expint,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error jy", is_show=True, is_legend=True)

        # plot atoms jz
        logger.info("Plotting data jz")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr]),
                                                                    np.array([lkf_num_history_manager.jzs,
                                                                              lkf_expint_approx_history_manager.jzs,
                                                                              lkf_exp_approx_history_manager.jzs,
                                                                              extended_kf_history_manager.jzs,
                                                                              unscented_kf_history_manager.jzs,
                                                                              j_z_full_history]),
                                                                    labels=labels,
                                                                    bools=[args[0].lkf_num,
                                                                           args[0].lkf_exp,
                                                                           args[0].lkf_expint,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Atoms jz", is_show=True, is_legend=True)

        # plot error for atoms jz
        logger.info("Plotting error jz")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter]),
                                                                    np.array([lkf_num_history_manager.jzs_err_post,
                                                                              lkf_expint_approx_history_manager.jzs_err_post,
                                                                              lkf_exp_approx_history_manager.jzs_err_post,
                                                                              extended_kf_history_manager.jzs_err_post,
                                                                              unscented_kf_history_manager.jzs_err_post,
                                                                              steady_state_history_manager.steady_posts_jz]),
                                                                    labels=labels_err,
                                                                    bools=[args[0].lkf_num,
                                                                           args[0].lkf_exp,
                                                                           args[0].lkf_expint,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error jz", is_show=True, is_legend=True)

        # plot light q (noisy, exact and filtered)
        logger.info("Plotting data q")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr]),
                                                                    np.array([lkf_num_history_manager.qs,
                                                                              lkf_expint_approx_history_manager.qs,
                                                                              lkf_exp_approx_history_manager.qs,
                                                                              extended_kf_history_manager.qs,
                                                                              unscented_kf_history_manager.qs,
                                                                              q_q_full_history]),
                                                                    labels=labels,
                                                                    bools=[args[0].lkf_num,
                                                                           args[0].lkf_exp,
                                                                           args[0].lkf_expint,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="q", is_show=True, is_legend=True)

        # plot error for light q
        logger.info("Plotting error q")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter]),
                                                                    np.array([lkf_num_history_manager.qs_err_post,
                                                                              lkf_expint_approx_history_manager.qs_err_post,
                                                                              lkf_exp_approx_history_manager.qs_err_post,
                                                                              extended_kf_history_manager.qs_err_post,
                                                                              unscented_kf_history_manager.qs_err_post,
                                                                              steady_state_history_manager.steady_posts_q]),
                                                                    labels=labels_err,
                                                                    bools=[args[0].lkf_num,
                                                                           args[0].lkf_exp,
                                                                           args[0].lkf_expint,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error q", is_show=True, is_legend=True)

        # plot light p (noisy, exact and filtered)
        logger.info("Plotting data p")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr]),
                                                                    np.array([lkf_num_history_manager.ps,
                                                                              lkf_expint_approx_history_manager.ps,
                                                                              lkf_exp_approx_history_manager.ps,
                                                                              extended_kf_history_manager.ps,
                                                                              unscented_kf_history_manager.ps,
                                                                              q_p_full_history]),
                                                                    labels=labels,
                                                                    bools=[args[0].lkf_num,
                                                                           args[0].lkf_exp,
                                                                           args[0].lkf_expint,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Atoms p", is_show=True, is_legend=True)

        # plot error for atoms p
        logger.info("Plotting error p")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter]),
                                                                    np.array([lkf_num_history_manager.ps_err_post,
                                                                              lkf_expint_approx_history_manager.ps_err_post,
                                                                              lkf_exp_approx_history_manager.ps_err_post,
                                                                              extended_kf_history_manager.ps_err_post,
                                                                              unscented_kf_history_manager.ps_err_post,
                                                                              steady_state_history_manager.steady_posts_p]),
                                                                    labels=labels_err,
                                                                    bools=[args[0].lkf_num,
                                                                           args[0].lkf_exp,
                                                                           args[0].lkf_expint,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Error p", is_show=True, is_legend=True)