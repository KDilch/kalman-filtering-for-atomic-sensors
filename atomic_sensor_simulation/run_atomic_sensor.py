#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import os

from atomic_sensor_simulation.sensor.atomic_sensor_measurement_model import AtomicSensorMeasurementModel
from kalman_filter.kalmanfilter import DD_KalmanFilter
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
from atomic_sensor_simulation.plot_kalman import plot_simulation_and_kalman

def run__atomic_sensor(queue):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__+'_PID_'+str(os.getpid()))
    logger.addHandler(logging.handlers.QueueHandler(queue))
    logger.info('Starting execution of atomic sensor simulation.')
    config, args = queue.get()[0]
    # atomic_state_lin_dynamics_manager = args.dynamics
    num_iter_simulation = 2 * (np.pi * config.simulation['number_periods'] /config.physical_parameters['larmour_freq']) / config.simulation['dt_simulation']
    print(num_iter_simulation)
    num_iter_measurement = np.int(np.floor_divide(num_iter_simulation, config.filter['measure_every_nth']))
    print(num_iter_measurement)
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
    sensor = AtomicSensorMeasurementModel(config, dt_filter)
    measurement_history_manager = AtomicSensorMeasurementHistoryManager(is_store_all=args.save_measurement_history)

    # KALMAN FILTER====================================================
    # Kalman Filter
    kf_dynamical_model = AtomicStateLinearDynamicsManager(config,
                                                          dt=config.simulation['dt_simulation'],
                                                          discretization_active=True,
                                                          is_model_time_invariant=False,
                                                          initial_time=0,
                                                          discrete_dt=dt_filter)
    kf_measurement_model = AtomicSensorMeasurementModel(config, dt_filter)
    kalman_filter = None
    kalman_filter_history_manager = AtomicSensorMeasurementHistoryManager(is_store_all=True)

    unscented_kalman_filter = None
    extended_kalman_filter = None
    # compute a steady state solution

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
            if measurement_history_manager.is_the_first_measurement():
                kalman_filter = DD_KalmanFilter(dynamical_model=kf_dynamical_model,
                                                measurement_model=kf_measurement_model,
                                                z0=measurement_outcome)
            kalman_filter.predict()
            kalman_filter.update(measurement_outcome)
            kalman_filter_history_manager.add_history_point(history_point=[time, kalman_filter.x])
    plot_simulation_and_kalman(simulation_data=simulation_history_manager, kalman_data=kalman_filter_history_manager, show=True)
    return 0

    # # FIND STEADY STATE SOLUTION
    # steady_state_history_manager = SteadyStateHistoryManager(num_iter_filter, config, time_arr_filter)
    # for index, time_filter in enumerate(time_arr_filter):
    #     steady_prior, steady_post = compute_steady_state_solution_for_atomic_sensor(t=time_filter,
    #                                                                                 F=eval_matrix_of_functions(
    #                                                                                     state.F_transition_matrix,
    #                                                                                     time_filter),
    #                                                                                 model=linear_kf_model,
    #                                                                                 config=config)
    #     logger.debug("Steady dynamical_model solution: predict_cov=%r,\n update_cov=%r" % (steady_prior, steady_post))
    #     steady_state_history_manager.add_entry(steady_prior, steady_post, index)

