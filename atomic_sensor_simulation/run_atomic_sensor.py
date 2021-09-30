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
from atomic_sensor_simulation.history_manager.atomic_sensor_steady_state import AtomicSensorSteadyStateHistoryManager
from atomic_sensor_simulation.history_manager.atomic_sensor_filter import KalmanFilterHistoryManager
from state_dynamics_manager.atomic_state_linear_dynamics_manager import AtomicStateLinearDynamicsManager, AtomicStateSquareWaveManager, AtomicStateSawtoothWaveManager, AtomicStateSinWaveManager
from atomic_sensor_simulation.history_manager.atomic_sensor_simulation_history_manager import AtomicSensorSimulationHistoryManager
from atomic_sensor_simulation.history_manager.atomic_sensor_measurement_history_manager import AtomicSensorMeasurementHistoryManager
from atomic_sensor_simulation.kalman_filter.steady_state_solver import AtomicSensorSteadyStateSolver
from atomic_sensor_simulation.plot_kalman import plot_simulation_and_kalman, plot_kalman_and_steady_state


def run__atomic_sensor(queue):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__+'_PID_'+str(os.getpid()))
    logger.addHandler(logging.handlers.QueueHandler(queue))
    logger.info('Starting execution of atomic sensor simulation.')
    config, args = queue.get()[0]
    # atomic_state_lin_dynamics_manager = args.dynamics
    num_iter_simulation = 2 * (np.pi * config.simulation['number_periods'] /config.physical_parameters['larmour_freq']) / config.simulation['dt_simulation']
    num_iter_filter = np.int(np.floor_divide(num_iter_simulation * config.simulation['dt_simulation'],
                                             config.filter['dt_filter']))

    measure_every_nth = np.int(np.floor_divide(num_iter_simulation, num_iter_filter)) # the assumtion is that dt_simulation is continous in comparison to dt_filter
    num_iter_measurement = np.int(np.floor_divide(num_iter_simulation, measure_every_nth))
    dt_filter = config.filter['dt_filter']

    time_arr_simulation = np.arange(0, num_iter_simulation * config.simulation['dt_simulation'], config.simulation['dt_simulation'])
    time_arr_filter = np.arange(0, num_iter_filter * config.filter['dt_filter'], config.filter['dt_filter'])

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
    steady_state_solver = None
    kalman_filter_history_manager = KalmanFilterHistoryManager()
    steady_state_history_manager = AtomicSensorSteadyStateHistoryManager()
    unscented_kalman_filter = None
    extended_kalman_filter = None

    # compute a steady state solution

    simulation_steps_counter = 0
    for time in time_arr_simulation:

        simulation_steps_counter += 1
        is_measurement_performed = simulation_steps_counter % measure_every_nth

        # SIMULATE THE DYNAMICS
        atomic_state_dynamics_manager.step(time)
        # UPDATE THE SIMULATION HISTORY
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
                steady_state_solver = AtomicSensorSteadyStateSolver(kalman_filter)

            kalman_filter.predict()
            kalman_filter.update(measurement_outcome)
            kalman_filter_history_manager.add_history_point(history_point=[time, kalman_filter.x, kalman_filter.P_prior, kalman_filter.P_post])
            # FIND STEADY STATE SOLUTION
            steady_state_solver.steady_state_solution_rotating_frame(time)
            logger.debug("Steady dynamical_model solution: predict_cov=%r,\n update_cov=%r" % (steady_state_solver.steady_prior, steady_state_solver.steady_post))
            steady_state_history_manager.add_history_point([time, [steady_state_solver.steady_prior, steady_state_solver.steady_post]])

    plot_simulation_and_kalman(simulation_data=simulation_history_manager, kalman_data=kalman_filter_history_manager, show=True)
    plot_kalman_and_steady_state(kalman_data=kalman_filter_history_manager, steady_state_data=steady_state_history_manager, show=True)
    return
