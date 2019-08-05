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
    simulation_parser.set_defaults(func=run__atomic_sensor)

    tests_parser = subparsers.add_parser('run-atomic-sensor', help='')
    tests_parser.set_defaults(func=run__atomic_sensor)

    position_speed_parser = subparsers.add_parser('run-position-speed', help='')
    position_speed_parser.set_defaults(func=run_position_speed)

    # parse some argument lists
    args = parser.parse_args()
    logger.info('Parsed input arguments %r' % stringify_namespace(args))
    args.func(args)
    logger.info('Ending execution of the application.')
    return 0


def run__atomic_sensor(*args):
    from atomic_sensor_simulation.state.atomic_state import AtomicSensorState
    from atomic_sensor_simulation.sensor.atomic_sensor import AtomicSensor
    from atomic_sensor_simulation.model.atomic_sensor_model import AtomicSensorModel
    from atomic_sensor_simulation.utilities import calculate_error, compute_squred_error_from_covariance, eval_matrix_of_functions
    from atomic_sensor_simulation.atomic_sensor_steady_state import compute_steady_state_solution_for_atomic_sensor

    # Logger for storing errors and logs in seprate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-atomic-sensor command.')

    # PARAMETERS=====================================================

    ## physical parameters
    larmour_freq = 6.  #6.
    spin_correlation_const = 0.33 #25  # 1/T2
    light_correlation_const = 0.2
    logger.info('Setting physical parameters to larmour_freq = %r, spin_correlation_const = %r, light_correlation_const=%r.' %
                (str(larmour_freq),
                 str(spin_correlation_const),
                 str(light_correlation_const)
                 )
                )

    #consts for coupling function -> amplitude*cos(omega*t)
    omega = 6.0 #\omega_p
    amplitude = 30. #30. #g_p
    phase_shift = 0.0 #rad

    #simulation parameters
    number_periods = 10.
    dt_sensor = 0.01
    num_iter_sensor = (2*np.pi*number_periods/larmour_freq)/dt_sensor
    logger.info('Setting simulation parameters to num_iter_sensor = %r, delta_t_sensor = %r, number_periods=%r.' %
                (str(num_iter_sensor),
                 str(dt_sensor),
                 str(number_periods)
                 )
                )

    #filter parameters
    dt_filter = 0.02
    num_iter_filter = np.int(np.floor_divide(num_iter_sensor*dt_sensor, dt_filter))
    every_nth_z = np.int(np.floor_divide(num_iter_sensor, num_iter_filter))
    logger.info('Setting filter parameters to num_iter_filter = %r, delta_t_filter = %r.' %
                (str(num_iter_filter),
                 str(dt_filter),
                 )
                )

    #initial conditions for the filter
    #TO DO

    # SIMULATING DYNAMICS=====================================================
    time_arr = np.arange(0, num_iter_sensor*dt_sensor, dt_sensor)
    time_arr_filter = np.arange(0, num_iter_filter*dt_filter, dt_filter)

    
    #initial conditions for simulation
    spin_y_initial_val = 2.
    spin_z_initial_val = 2.
    quadrature_q_initial_val = 0.
    quadrature_p_initial_val = 0.
    logger.info('Setting initial state vec to  [%r, %r, %r, %r].' %
                (str(spin_y_initial_val),
                 str(spin_z_initial_val),
                 str(quadrature_q_initial_val),
                 str(quadrature_p_initial_val)
                 )
                )

    #noise and measurement strengths
    QJy = 0.1
    QJz = 0.1
    Qq = 0.05
    Qp = 0.02

    gD = 100
    QD = 0.01

    #Q, H and R definitions
    Q = np.array([[QJy, 0., 0., 0.],
                  [0., QJz, 0., 0.],
                  [0., 0., Qq, 0.],
                  [0., 0., 0., Qp]])

    H = np.array([[0., gD, 0., 0.]])
    R = np.array([[QD]])

    #W definition
    W_jy = np.array([[1., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.]])
    W_jz = np.array([[0., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.]])
    W_q = np.array([[0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 0.]])
    W_p = np.array([[0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 1.]])



    logger.info('Setting Q, H and R to Q = %r, H = %r, R = %r' %
                (str(Q),
                 str(H),
                 str(R)
                 )
                )

    state = AtomicSensorState(initial_vec=np.array([spin_y_initial_val, spin_z_initial_val, quadrature_q_initial_val, quadrature_p_initial_val]),
                              noise_vec=GaussianWhiteNoise(mean=[0., 0., 0., 0.], cov=Q, dt=dt_sensor),
                              initial_time=0,
                              dt=dt_sensor,
                              light_correlation_const=light_correlation_const,
                              spin_correlation_const=spin_correlation_const,
                              larmour_freq=larmour_freq,
                              coupling_amplitude=amplitude,
                              coupling_freq=omega,
                              coupling_phase_shift=phase_shift)

    sensor = AtomicSensor(state,
                          sensor_noise=GaussianWhiteNoise(mean=0., cov=R/dt_sensor, dt=dt_sensor),
                          H=H,
                          dt=dt_sensor)

    zs = np.array([np.array((sensor.read(_))) for _ in time_arr])  # noisy measurement
    zs_filter_freq = zs[::every_nth_z]
    x_filter_freq = sensor.state_vec_full_history[::every_nth_z]

    # KALMAN FILTER====================================================
    #Definning dynamical equations for the filter
    model = AtomicSensorModel(F=state.F_transition_matrix,
                              Q=Q,
                              H=H,
                              R=R/dt_filter,
                              Gamma=state.Gamma_control_evolution_matrix,
                              u=state.u_control_vec,
                              z0=[zs[0]],
                              dt=dt_filter)

    # RUN FILTERPY KALMAN FILTER
    logger.info("Initializing filterpy Kalman Filter")
    filterpy = model.initialize_filterpy()
    filtered_light_p = np.zeros(num_iter_filter)
    filtered_atoms_jy = np.zeros(num_iter_filter)
    filtered_light_q = np.zeros(num_iter_filter)
    filtered_atoms_jz = np.zeros(num_iter_filter)
    error_jy = np.zeros(num_iter_filter)
    error_jz = np.zeros(num_iter_filter)
    error_q = np.zeros(num_iter_filter)
    error_p = np.zeros(num_iter_filter)
    filter_error_jy_prior = np.zeros(num_iter_filter)
    filter_error_jz_prior = np.zeros(num_iter_filter)
    filter_error_q_prior = np.zeros(num_iter_filter)
    filter_error_p_prior = np.zeros(num_iter_filter)
    filter_error_jy_post = np.zeros(num_iter_filter)
    filter_error_jz_post = np.zeros(num_iter_filter)
    filter_error_q_post = np.zeros(num_iter_filter)
    filter_error_p_post = np.zeros(num_iter_filter)

    for index, time in enumerate(time_arr_filter):
        z = zs_filter_freq[index]
        filterpy.predict(F=model.compute_Phi_delta(from_time=time-dt_filter))
        filterpy.update(z)
        filtered_atoms_jy[index] = filterpy.x[0]
        filtered_atoms_jz[index] = filterpy.x[1]
        filtered_light_q[index] = filterpy.x[2]
        filtered_light_p[index] = filterpy.x[3]
        error_jy[index] = calculate_error(W_jy, x=x_filter_freq[index], x_est=filterpy.x)
        error_jz[index] = calculate_error(W_jz, x=x_filter_freq[index], x_est=filterpy.x)
        error_q[index] = calculate_error(W_q, x=x_filter_freq[index], x_est=filterpy.x)
        error_p[index] = calculate_error(W_p, x=x_filter_freq[index], x_est=filterpy.x)
        filter_error_jy_prior[index] = compute_squred_error_from_covariance(filterpy.P_prior, index=0)
        filter_error_jz_prior[index] = compute_squred_error_from_covariance(filterpy.P_prior, index=1)
        filter_error_q_prior[index] = compute_squred_error_from_covariance(filterpy.P_prior, index=2)
        filter_error_p_prior[index] = compute_squred_error_from_covariance(filterpy.P_prior, index=3)
        filter_error_jy_post[index] = compute_squred_error_from_covariance(filterpy.P_post, index=0)
        filter_error_jz_post[index] = compute_squred_error_from_covariance(filterpy.P_post, index=1)
        filter_error_q_post[index] = compute_squred_error_from_covariance(filterpy.P_post, index=2)
        filter_error_p_post[index] = compute_squred_error_from_covariance(filterpy.P_post, index=3)

    # # RUN HOMEMADE KALMAN FILTER
    # logger.info("Initializing homemade Kalman Filter")
    # home_made_kf = model.initialize_homemade_filter()
    #
    # filtered_light_p_homemade = np.zeros(num_iter_filter)
    # filtered_atoms_jy_homemade = np.zeros(num_iter_filter)
    # filtered_light_q_homemade = np.zeros(num_iter_filter)
    # filtered_atoms_jz_homemade = np.zeros(num_iter_filter)
    # for index, time in enumerate(time_arr_filter):
    #     z = zs_filter_freq[index]
    #     home_made_kf.predict(from_time=time, to_time=time+dt_sensor, Phi_delta=model.compute_Phi_delta(from_time=time-dt_sensor))
    #     home_made_kf.update(z)
    #     filtered_atoms_jy_homemade[index] = filterpy.x[0]
    #     filtered_atoms_jz_homemade[index] = filterpy.x[1]
    #     filtered_light_q_homemade[index] = filterpy.x[2]
    #     filtered_light_p_homemade[index] = filterpy.x[3]


    #FIND STEADY STATE SOLUTION
    steady_prior, steady_post = compute_steady_state_solution_for_atomic_sensor(coupling_freq=omega,
                                                                                coupling_phase_shift=phase_shift,
                                                                                t=0.,
                                                                                F=eval_matrix_of_functions(state.F_transition_matrix, 0.),
                                                                                model=model)
    logger.info("Steady state solution: predict_cov=%r,\n update_cov=%r" % (steady_prior, steady_post))
    # PLOTS=========================================================
    # Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_q_full_history, q_p_full_history = zip(*sensor.state_vec_full_history)

    #plot atoms jy
    logger.info("Plotting data jy")
    plt.title("Atoms jy")
    # plt.plot(time_arr_filter, filtered_atoms_jy_homemade, label='Homemade')
    plt.plot(time_arr_filter, filtered_atoms_jy, label='Filterpy')
    plt.plot(time_arr, j_y_full_history, label='Exact data')
    plt.legend()
    plt.show()

    # plot error for atoms jy
    logger.info("Plotting error jy")
    plt.title("Squared error jy")
    plt.plot(time_arr_filter, filter_error_jy_prior, label='Prior')
    plt.plot(time_arr_filter, filter_error_jy_post, label='Post')
    # plt.plot(time_arr_filter, error_jy, label='Filterpy')
    plt.axhline(y=steady_post[0][0], color='r', linestyle='-', label='steady_post')
    plt.axhline(y=steady_prior[0][0], color='b', linestyle='-', label='steady_prior')
    plt.legend()
    plt.show()

    #plot atoms jz
    logger.info("Plotting data jz")
    plt.title("Atoms jz")
    # plt.plot(time_arr_filter, filtered_atoms_jz_homemade, label='Homemade')
    plt.plot(time_arr_filter, filtered_atoms_jz, label='Filterpy')
    plt.plot(time_arr, j_z_full_history, label='Exact data')
    plt.legend()
    plt.show()

    # plot error for atoms jz
    logger.info("Plotting error jz")
    plt.title("Squared error jz")
    plt.plot(time_arr_filter, filter_error_jz_prior, label='Prior')
    plt.plot(time_arr_filter, filter_error_jz_post, label='Post')
    # plt.plot(time_arr_filter, error_jz, label='Squared error jz')
    plt.axhline(y=steady_post[1][1], color='r', linestyle='-', label='steady_post')
    plt.axhline(y=steady_prior[1][1], color='b', linestyle='-', label='steady_prior')
    plt.legend()
    plt.show()

    # plot light q (noisy, exact and filtered)
    logger.info("Plotting data")
    plt.title("Light q")
    # plt.plot(time_arr_filter, filtered_light_q_homemade, label='Homemade')
    plt.plot(time_arr_filter, filtered_light_q, label='Filterpy')
    plt.plot(time_arr, q_q_full_history, label='Exact data')
    plt.legend()
    plt.show()

    # plot error q
    logger.info("Plotting error q")
    plt.title("Squared error q")
    plt.plot(time_arr_filter, filter_error_q_prior, label='Prior')
    plt.plot(time_arr_filter, filter_error_q_post, label='Post')
    # plt.plot(time_arr_filter, error_q, label='Filterpy')
    plt.axhline(y=steady_post[2][2], color='r', linestyle='-', label='steady_post')
    plt.axhline(y=steady_prior[2][2], color='b', linestyle='-', label='steady_prior')
    plt.legend()
    plt.show()

    # # plot light p (noisy, exact and filtered)
    # logger.info("Plotting data")
    # plt.title("Light p")
    # # plt.plot(time_arr_filter, filtered_light_p_homemade, label='Homemade')
    # plt.plot(time_arr_filter, filtered_light_p, label='Filterpy')
    # plt.plot(time_arr, q_p_full_history, label='Exact data')
    # plt.legend()
    # plt.show()

    # # plot error p
    # logger.info("Plotting error p")
    # plt.title("Squared error p")
    # plt.plot(time_arr_filter, filter_error_p_prior, label='Filter error prior')
    # plt.plot(time_arr_filter, filter_error_p_post, label='Filter error post')
    # plt.plot(time_arr_filter, error_p, label='Filterpy')
    # plt.legend()
    # plt.show()

    # # plot zs
    # plt.plot(time_arr, sensor.z_no_noise_arr, label='Exact sensor data')
    # plt.plot(time_arr, zs, label='Noisy sensor readings')
    # plt.legend()
    # plt.show()


def run_position_speed(*args):

    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run_position_speed command.')
    dt = 0.5
    num_iter = 20

    from atomic_sensor_simulation.state.pos_vel_state import PosVelSensorState
    from atomic_sensor_simulation.model.pos_vel_model import PosVelModel
    from atomic_sensor_simulation.sensor import pos_sensor

    H = np.array([[1. / 0.3048, 0., 0., 0.], [0., 0., 1. / 0.3048, 0.]], dtype='float64')
    R = np.eye(2) * 5.
    Q = np.array([[0.05, 0., 0., 0.],
                 [0., 0.05, 0., 0.],
                 [0., 0., 0.05, 0.],
                 [0., 0., 0., 0.05]])

    state = PosVelSensorState(initial_vec=np.array([0., 0., 2., 1.]),
                              noise_vec=GaussianWhiteNoise(mean=[0., 0., 0., 0.], cov=Q, dt=dt),
                              initial_time=0,
                              dt=dt)

    sensor = pos_sensor.PosSensor(state,
                                  scalar_strenght_y=1.,
                                  dt=dt)
    time_arr = np.arange(0, num_iter, dt)

    zs = np.array([np.array((sensor.read(_))) for _ in time_arr])

    # KALMAN FILTER============================================================================
    model = PosVelModel(state.F_transition_matrix,
                        Gamma=state.Gamma_control_evolution_matrix,
                        u=state.u_control_vec,
                        z0=zs[0],
                        dt=dt)

    filterpy_kf = model.initialize_filterpy()
    (mu, cov, _, _) = filterpy_kf.batch_filter(zs)
    zs *= .3048

    # plot position
    plt.plot(mu[:, 0], mu[:, 2], label='Filtered signal')
    plt.plot(zs[:, 0], zs[:, 1], label='Noisy signal')
    plt.legend()
    plt.show()


def run_tests(*args):
    #:TODO implement
    pass


if __name__ == "__main__":

    main()
