#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np

from atomic_sensor_simulation.utilities import import_config_from_path
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.state.atomic_state import AtomicSensorState
from atomic_sensor_simulation.sensor.atomic_sensor import AtomicSensor
from filter_model.AtomicSensor.linear_kf import Linear_KF
from filter_model.AtomicSensor.unscented_kf import Unscented_KF
from filter_model.AtomicSensor.extented_kf import Extended_KF
from atomic_sensor_simulation.utilities import calculate_error, eval_matrix_of_functions
from atomic_sensor_simulation.helper_functions.plot_all_atomic_sensor import plot__all_atomic_sensor
from atomic_sensor_simulation.helper_functions.save_all_simulation_data import save_data
from history_manager.atomic_sensor_history_manager import Filter_History_Manager, SteadyStateHistoryManager
from atomic_sensor_simulation.atomic_sensor_steady_state import compute_steady_state_solution_for_atomic_sensor


def run__atomic_sensor(*args):

    # Logger for storing errors and logs in seprate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-atomic-sensor command.')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    config.coupling['g_p'] = args[0].gp
    config.coupling['omega_p'] = args[0].wp

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
                                                           cov=Q*config.simulation['dt_sensor'],
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
    logger.info('Filter frequency is  %r' % every_nth_z)
    zs_filter_freq = zs[::every_nth_z]
    x_filter_freq = sensor.state_vec_full_history[::every_nth_z]
    waveform_filter_freq = sensor.waveform_history[::every_nth_z]

    zs_sigma = zs_filter_freq/np.sqrt(R/config.filter['dt_filter'])


    # KALMAN FILTER====================================================
    # Linear Kalman Filter
    linear_kf_model = Linear_KF(F=state.F_transition_matrix,
                                Q=Q,
                                H=H,
                                R=R,
                                R_delta=R / config.filter['dt_filter'],
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

    unscented_kf_model = Unscented_KF(F=state.F_transition_matrix,
                                      Q=linear_kf_model.Q_delta,
                                      R=R,
                                      H=H,
                                      R_delta=R / config.filter['dt_filter'],
                                      Gamma=state.Gamma_control_evolution_matrix,
                                      u=state.u_control_vec,
                                      z0=[zs[0]],
                                      dt=config.filter['dt_filter'],
                                      x0=linear_kf_model.x0,
                                      P0=linear_kf_model.P0)

    extended_kf_model = Extended_KF(F=state.F_transition_matrix,
                                    H=H,
                                    Q=Q,
                                    R=R,
                                    R_delta=R / config.filter['dt_filter'],
                                    Gamma=state.Gamma_control_evolution_matrix,
                                    u=state.u_control_vec,
                                    z0=[zs[0]],
                                    dt=config.filter['dt_filter'],
                                    x0=linear_kf_model.x0,
                                    P0=linear_kf_model.P0,
                                    num_terms=3,
                                    time_arr=time_arr_filter
                                    )

    extended_kf_model_lin = Extended_KF(F=state.F_transition_matrix,
                                    H=H,
                                    Q=Q,
                                    R=R,
                                    R_delta=R / config.filter['dt_filter'],
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
    lkf_exp_approx_history_manager = Filter_History_Manager(lkf_exp_approx, num_iter_filter, config, time_arr_filter)

    lkf_expint_approx = linear_kf_model.initialize_filterpy()
    lkf_expint_approx_history_manager = Filter_History_Manager(lkf_expint_approx, num_iter_filter, config, time_arr_filter)

    lkf_num = linear_kf_model.initialize_filterpy()
    lkf_num_history_manager = Filter_History_Manager(lkf_num, num_iter_filter, config, time_arr_filter)

    logger.info("Initializing unscented_kf_filterpy Unscented Filter")
    unscented_kf_filterpy = unscented_kf_model.initialize_filterpy()
    unscented_kf_history_manager = Filter_History_Manager(unscented_kf_filterpy, num_iter_filter, config, time_arr_filter)

    logger.info("Initializing extended_kf_filterpy Unscented Filter")
    extended_kf_filterpy = extended_kf_model.initialize_filterpy(
        light_correlation_const=config.physical_parameters['light_correlation_const'],
        spin_correlation_const=config.physical_parameters['spin_correlation_const'],
        larmour_freq=config.physical_parameters['larmour_freq'],
        coupling_amplitude=config.coupling['g_p'],
        coupling_freq=config.coupling['omega_p'],
        coupling_phase_shift=config.coupling['phase_shift'])
    extended_kf_filterpy_lin = extended_kf_model.initialize_filterpy(
        light_correlation_const=config.physical_parameters['light_correlation_const'],
        spin_correlation_const=config.physical_parameters['spin_correlation_const'],
        larmour_freq=config.physical_parameters['larmour_freq'],
        coupling_amplitude=config.coupling['g_p'],
        coupling_freq=config.coupling['omega_p'],
        coupling_phase_shift=config.coupling['phase_shift'])
    extended_kf_history_manager = Filter_History_Manager(extended_kf_filterpy, num_iter_filter, config, time_arr_filter)
    extended_kf_history_manager_lin = Filter_History_Manager(extended_kf_filterpy_lin, num_iter_filter, config, time_arr_filter)

    error_jy_LKF = np.zeros(num_iter_filter)
    error_jz_LKF = np.zeros(num_iter_filter)
    error_q_LKF = np.zeros(num_iter_filter)
    error_p_LKF = np.zeros(num_iter_filter)
    error_jy_EKF = np.zeros(num_iter_filter)
    error_jz_EKF = np.zeros(num_iter_filter)
    error_q_EKF = np.zeros(num_iter_filter)
    error_p_EKF = np.zeros(num_iter_filter)
    error_waveform_LKF = np.zeros(num_iter_filter)
    error_waveform_EKF = np.zeros(num_iter_filter)

    for index, time in enumerate(time_arr_filter):
        z = zs_filter_freq[index]

        extended_kf_filterpy_lin.set_Q(lkf_num.Q) # THIS LINE NEEDS TO BE ADDED IF LINEARIZATION FIRST
        extended_kf_filterpy_lin.predict()
        extended_kf_filterpy_lin.update(z,
                                    extended_kf_model_lin.HJacobianat,
                                    extended_kf_model_lin.hx,
                                    R=extended_kf_model_lin.R_delta)
        extended_kf_history_manager_lin.add_entry(index)

        extended_kf_filterpy.predict_discretization_first()
        extended_kf_filterpy.update(z,
                                    extended_kf_model.HJacobianat,
                                    extended_kf_model.hx,
                                    R=extended_kf_filterpy.R_delta)
        extended_kf_history_manager.add_entry(index)

        #COMMENTING OUT UKF SINCE IT NEEDS DEBUGGING (no reason to wait longer for the simulation to finish)
        # unscented_kf_model.set_Q(Q=lkf_num.Q)
        # unscented_kf_filterpy.predict(fx=unscented_kf_model.compute_fx_at_time_t(time))
        # unscented_kf_filterpy.update(z)
        # unscented_kf_history_manager.add_entry(index)

        lkf_num.predict()
        lkf_num.F = linear_kf_model.compute_Phi_delta_solve_ode_numerically(from_time=time)
        lkf_num.Q = linear_kf_model.compute_Q_delta_sympy(from_time=time,
                                                          Phi_0=lkf_num.F,
                                                          num_terms=30)
        lkf_expint_approx.predict()
        lkf_expint_approx.F = linear_kf_model.compute_Phi_delta_solve_ode_numerically(from_time=time)
        lkf_exp_approx.predict()
        lkf_exp_approx.F = linear_kf_model.compute_Phi_delta_solve_ode_numerically(from_time=time)
        # logger.debug('Setting Phi to [%r]' % str(lkf_num.F))
        lkf_num.update(z)
        # lkf_expint_approx.update(z)
        lkf_exp_approx.update(z)

        lkf_num_history_manager.add_entry(index)
        # lkf_expint_approx_history_manager.add_entry(index)
        lkf_exp_approx_history_manager.add_entry(index)

        error_jy_LKF[index] = calculate_error(config.W['W_jy'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_jz_LKF[index] = calculate_error(config.W['W_jz'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_q_LKF[index] = calculate_error(config.W['W_q'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_p_LKF[index] = calculate_error(config.W['W_p'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_jy_EKF[index] = calculate_error(config.W['W_jy'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_jz_EKF[index] = calculate_error(config.W['W_jz'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_q_EKF[index] = calculate_error(config.W['W_q'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_p_EKF[index] = calculate_error(config.W['W_p'], x=x_filter_freq[index], x_est=lkf_num.x)
        error_waveform_LKF[index] = (waveform_filter_freq[index]-lkf_num_history_manager.waveform_est[index])**2
        error_waveform_EKF[index] = config.coupling['g_p']*(np.cos(config.coupling['omega_p']*time)**2*error_q_EKF[index]+np.sin(config.coupling['omega_p']*time)**2*error_p_EKF[index])

    # FIND STEADY STATE SOLUTION
    steady_state_history_manager = SteadyStateHistoryManager(num_iter_filter, config, time_arr_filter)
    for index, time_filter in enumerate(time_arr_filter):
        steady_prior, steady_post = compute_steady_state_solution_for_atomic_sensor(t=time_filter,
                                                                                    F=eval_matrix_of_functions(
                                                                                        state.F_transition_matrix,
                                                                                        time_filter),
                                                                                    model=linear_kf_model)
        logger.debug("Steady state solution: predict_cov=%r,\n update_cov=%r" % (steady_prior, steady_post))
        steady_state_history_manager.add_entry(steady_prior, steady_post, index)

    # # # PLOT DATA
    plot__all_atomic_sensor(sensor,
                            time_arr_filter,
                            time_arr,
                            lkf_num_history_manager,
                            lkf_expint_approx_history_manager,
                            lkf_exp_approx_history_manager,
                            extended_kf_history_manager,
                            extended_kf_history_manager_lin,
                            unscented_kf_history_manager,
                            steady_state_history_manager,
                            np.transpose(zs_filter_freq[:-1])[0],
                            error_jy_LKF,
                            error_jz_LKF,
                            error_q_LKF,
                            error_p_LKF,
                            error_waveform_LKF,
                            error_jy_EKF,
                            error_jz_EKF,
                            error_q_EKF,
                            error_p_EKF,
                            error_waveform_EKF,
                            np.transpose(zs_sigma[:-1])[0],
                            args,
                            config)

    # SAVE DATA TO A FILE
    save_data(sensor,
              time_arr_filter,
              time_arr,
              lkf_num_history_manager,
              lkf_exp_approx_history_manager,
              extended_kf_history_manager,
              extended_kf_history_manager_lin,
              unscented_kf_history_manager,
              steady_state_history_manager,
              np.transpose(zs_filter_freq[:-1])[0],
              error_jy_LKF,
              error_jz_LKF,
              error_q_LKF,
              error_p_LKF,
              error_waveform_LKF,
              error_jy_EKF,
              error_jz_EKF,
              error_q_EKF,
              error_p_EKF,
              error_waveform_EKF,
              np.transpose(zs_sigma[:-1])[0],
              args,
              './Simulation_data/data',
              config)
