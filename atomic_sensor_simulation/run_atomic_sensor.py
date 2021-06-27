#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import os

from atomic_sensor_simulation.sensor.atomic_sensor_measurement_model import AtomicSensorMeasurementModel
from kalman_filter.kalmanfilter import KalmanFilter
from kalman_filter.AtomicSensor.unscented_kf import Unscented_KF
from kalman_filter.AtomicSensor.extented_kf import Extended_KF
from atomic_sensor_simulation.utilities import calculate_error, eval_matrix_of_functions
from atomic_sensor_simulation.helper_functions.plot_all_atomic_sensor import plot__all_atomic_sensor
from atomic_sensor_simulation.helper_functions.save_all_simulation_data import save_data
from history_manager.atomic_sensor_history_manager import Filter_History_Manager, SteadyStateHistoryManager
from atomic_sensor_simulation.atomic_sensor_steady_state import compute_steady_state_solution_for_atomic_sensor
from state_dynamics_manager.atomic_state_linear_dynamics_manager import AtomicStateLinearDynamicsManager, AtomicStateSquareWaveManager, AtomicStateSawtoothWaveManager, AtomicStateSinWaveManager
from atomic_sensor_simulation.history_manager.atomic_sensor_simulation_history_manager import AtomicSensorSimulationHistoryManager
from atomic_sensor_simulation.history_manager.atomic_sensor_measurement_history_manager import AtomicSensorMeasurementHistoryManager

def run__atomic_sensor(queue):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__+'_PID_'+str(os.getpid()))
    logger.addHandler(logging.handlers.QueueHandler(queue))
    logger.info('Starting execution of atomic sensor simulation.')
    config, args = queue.get()[0]
    # atomic_state_lin_dynamics_manager = args.dynamics
    num_iter_simulation = (2 * np.pi * config.simulation['number_periods'] /
                           config.physical_parameters['larmour_freq']) / config.simulation['dt_simulation']
    num_iter_measurement = np.int(np.floor_divide(num_iter_simulation, config.filter['measure_every_nth']))
    dt_filter = num_iter_simulation*config.simulation['dt_simulation']/num_iter_measurement

    time_arr_simulation = np.arange(0, num_iter_simulation * config.simulation['dt_simulation'], config.simulation['dt_simulation'])
    time_arr_measurement = np.arange(0, num_iter_measurement * dt_filter, dt_filter)

    # SIMULATE THE DYNAMICS AND PERFORM A REAL TIME MEASUREMENT FOR EVERY NTH SIMULATED VALUE===========================

    # INITIALIZE THE SIMULATION
    # There is a StateDynamicsManager for every case of dynamical model considered for an atomic sensor
    if config.simulation['simulation_type'] == 'sin':
        atomic_state_dynamics_manager = AtomicStateSinWaveManager(config)
    if config.simulation['simulation_type'] == 'square':
        atomic_state_dynamics_manager = AtomicStateSquareWaveManager(config)
    if config.simulation['simulation_type'] == 'sawtooth':
        atomic_state_dynamics_manager = AtomicStateSawtoothWaveManager(config, dt=config.simulation['dt_simulation'])
    if config.simulation['simulation_type'] == 'linear':
        atomic_state_dynamics_manager = AtomicStateLinearDynamicsManager(config, dt=config.simulation['dt_simulation'])
    else:
        raise ValueError('Invalid simulation type %r.' % config.simulation['simulation_type'])

    simulation_history_manager = AtomicSensorSimulationHistoryManager()

    # INITIALIZE THE MEASUREMENT
    sensor = AtomicSensorMeasurementModel(config)
    measurement_history_manager = AtomicSensorMeasurementHistoryManager(is_store_all=args.save_measurement_history)
    simulation_steps_counter = 0
    for time in time_arr_simulation:
        simulation_steps_counter += 1
        # SIMULATE THE DYNAMICS
        atomic_state_dynamics_manager.step(time)
        # UPDATE THE SIMULATION HISTORY
        is_measurement_performed = simulation_steps_counter % config.filter['measure_every_nth']
        simulation_history_manager.add_history_point(history_point=[time, atomic_state_dynamics_manager.vec],
                                                     is_measurement_performed=is_measurement_performed)

        # IF it's THE nth SIMULATED VALUE - PERFORM A MEASUREMENT
        if is_measurement_performed:
            measurement_outcome = sensor.read(state_vec=atomic_state_dynamics_manager.vec)
            measurement_history_manager.add_history_point(history_point=[time, measurement_outcome])

    # KALMAN FILTER====================================================
    # Kalman Filter
    kf_dynamical_model = None
    kf_measurement_model = None
    kalman_filter = None
    unscented_kalman_filter = None
    extended_kalman_filter = None


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
        error_jy_EKF[index] = calculate_error(config.W['W_jy'], x=x_filter_freq[index], x_est=extended_kf_filterpy.x)
        error_jz_EKF[index] = calculate_error(config.W['W_jz'], x=x_filter_freq[index], x_est=extended_kf_filterpy.x)
        error_q_EKF[index] = calculate_error(config.W['W_q'], x=x_filter_freq[index], x_est=extended_kf_filterpy.x)
        error_p_EKF[index] = calculate_error(config.W['W_p'], x=x_filter_freq[index], x_est=extended_kf_filterpy.x)
        error_waveform_LKF[index] = (waveform_filter_freq[index]-lkf_num_history_manager.waveform_est[index])**2
        error_waveform_EKF[index] = (waveform_filter_freq[index]-extended_kf_history_manager.waveform_est[index])**2

    # FIND STEADY STATE SOLUTION
    steady_state_history_manager = SteadyStateHistoryManager(num_iter_filter, config, time_arr_filter)
    for index, time_filter in enumerate(time_arr_filter):
        steady_prior, steady_post = compute_steady_state_solution_for_atomic_sensor(t=time_filter,
                                                                                    F=eval_matrix_of_functions(
                                                                                        state.F_transition_matrix,
                                                                                        time_filter),
                                                                                    model=linear_kf_model,
                                                                                    config=config)
        logger.debug("Steady dynamical_model solution: predict_cov=%r,\n update_cov=%r" % (steady_prior, steady_post))
        steady_state_history_manager.add_entry(steady_prior, steady_post, index)

    # # PLOT DATA #TODO make process safe
    plot__all_atomic_sensor(sensor,
                            time_arr_filter,
                            time_arr_simulation,
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
              time_arr_simulation,
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
    return 1
