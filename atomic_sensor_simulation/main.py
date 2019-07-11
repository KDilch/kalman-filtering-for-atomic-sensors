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

    # Logger for storing errors and logs in seprate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-atomic-sensor command.')

    # PARAMETERS=====================================================

    ## physical parameters
    #Larmour?
    lamour_freq = 6.
    #TODO: change name to spins_...
    atoms_correlation_const = 0.5 # 1/T2
    light_correlation_const = 0.3333
    g_d_COUPLING_CONST = 100.
    SCALAR_STREGTH_z = 1.
    SCALAR_STREGTH_jy = 1.
    SCALAR_STREGTH_jz = 1.
    SCALAR_STRENGTH_qp = 1.0
    SCALAR_STRENGTH_qq = 1.0

    #consts for coupling function -> amplitude*cos(omega*t)
    omega = 0.
    amplitude = 0.0

    #simulation parameters
    number_periods = 0.5
    dt_sensor = 0.005
    num_iter_sensor = (2*np.pi*number_periods/lamour_freq)/dt_sensor
    print(num_iter_sensor)
    #num_iter_sensor = 5000
    
    #filter parameters
    dt_filter = 0.01
    num_iter_filter = np.int(np.floor_divide(num_iter_sensor*dt_sensor, dt_filter))
    every_nth_z = np.int(np.floor_divide(num_iter_sensor, num_iter_filter))


    # SIMULATING DYNAMICS=====================================================

    logger.info('Setting simulation parameters to num_iter_sensor = %r, delta_t_sensor = %r, light_correlation_const=%r.' %
                (str(num_iter_sensor),
                 str(dt_sensor),
                 str(light_correlation_const)
                 )
                )
    
    time_arr = np.arange(0, num_iter_sensor*dt_sensor, dt_sensor)
    time_arr_filter = np.arange(0, num_iter_filter*dt_filter, dt_filter)

    
    #initial conditions
    spin_y_initial_val = 1.
    spin_z_initial_val = 1.
    quadrature_p_initial_val = 1.
    quadrature_q_initial_val = 1.

    #TODO: Now should be definitions of global Q and H

    logger.info('Setting initial conditions to spin = [%r, %r], quadrature = [%r, %r]' %
                (str(spin_y_initial_val),
                 str(spin_z_initial_val),
                 str(quadrature_p_initial_val),
                 str(quadrature_q_initial_val)
                 )
                )

    state = AtomicSensorState(initial_vec=np.array([spin_y_initial_val, spin_z_initial_val, quadrature_p_initial_val, quadrature_q_initial_val]),
                              noise_vec=np.array([GaussianWhiteNoise(spin_y_initial_val,
                                                                     scalar_strength=SCALAR_STREGTH_jy,
                                                                     dt=dt_sensor),
                                                  GaussianWhiteNoise(spin_z_initial_val,
                                                                     scalar_strength=SCALAR_STREGTH_jz,
                                                                     dt=dt_sensor),
                                                  GaussianWhiteNoise(quadrature_q_initial_val,
                                                                     scalar_strength=SCALAR_STRENGTH_qp,
                                                                     dt=dt_sensor),
                                                  GaussianWhiteNoise(quadrature_q_initial_val,
                                                                     scalar_strength=SCALAR_STRENGTH_qq,
                                                                     dt=dt_sensor)
                                                  ]),
                              initial_time=0,
                              dt=dt_sensor,
                              light_correlation_const=light_correlation_const,
                              spin_correlation_const=atoms_correlation_const,
                              lamour_freq=lamour_freq,
                              coupling_amplitude=amplitude,
                              coupling_freq=omega)

    sensor = AtomicSensor(state,
                          scalar_strenght_z=SCALAR_STREGTH_z,
                          g_d_COUPLING_CONST=g_d_COUPLING_CONST,
                          dt=dt_sensor)

    zs = np.array([np.array((sensor.read(_))) for _ in time_arr])  # noisy measurement
    zs_filter_freq = zs[::every_nth_z]

    # KALMAN FILTER====================================================
    #Definning dynamical equations for the filter
    model = AtomicSensorModel(F=state.F_transition_matrix,
                              Gamma=state.Gamma_control_evolution_matrix,
                              u=state.u_control_vec,
                              z0=[zs[0]],
                              scalar_strength_z=SCALAR_STREGTH_z,
                              scalar_strength_jy=SCALAR_STREGTH_jy,
                              scalar_strength_jz=SCALAR_STREGTH_jz,
                              scalar_strength_qp=SCALAR_STRENGTH_qp,
                              scalar_strength_qq=SCALAR_STRENGTH_qq,
                              g_d_COUPLING_CONST=g_d_COUPLING_CONST,
                              dt=dt_filter)

    # RUN FILTERPY KALMAN FILTER
    logger.info("Initializing filterpy Kalman Filter")
    filterpy = model.initialize_filterpy()
    filtered_light_p = np.zeros(num_iter_filter)
    filtered_atoms_jy = np.zeros(num_iter_filter)
    filtered_light_q = np.zeros(num_iter_filter)
    filtered_atoms_jz = np.zeros(num_iter_filter)
    for index, time in enumerate(time_arr_filter):
        z = zs_filter_freq[index]
        filterpy.predict(F=model.compute_Phi_delta(from_time=time-dt_filter))
        filterpy.update(z)
        filtered_light_q[index] = filterpy.x[2]
        filtered_light_p[index] = filterpy.x[3]
        filtered_atoms_jy[index] = filterpy.x[0]
        filtered_atoms_jz[index] = filterpy.x[1]

    # RUN HOMEMADE KALMAN FILTER
    logger.info("Initializing homemade Kalman Filter")
    home_made_kf = model.initialize_homemade_filter()
    logger.info("Steady state solution: predict_cov=%r,\n update_cov=%r" % (home_made_kf.steady_state()))
    filtered_light_p_homemade = np.zeros(num_iter_filter)
    filtered_atoms_jy_homemade = np.zeros(num_iter_filter)
    filtered_light_q_homemade = np.zeros(num_iter_filter)
    filtered_atoms_jz_homemade = np.zeros(num_iter_filter)
    for index, time in enumerate(time_arr_filter):
        z = zs_filter_freq[index]
        home_made_kf.predict(from_time=time, to_time=time+dt_sensor, Phi_delta=model.compute_Phi_delta(from_time=time-dt_sensor))
        home_made_kf.update(z)
        filtered_light_q_homemade[index] = filterpy.x[2]
        filtered_light_p_homemade[index] = filterpy.x[3]
        filtered_atoms_jy_homemade[index] = filterpy.x[0]
        filtered_atoms_jz_homemade[index] = filterpy.x[1]


    # PLOTS=========================================================
    #Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_p_full_history, q_q_full_history = zip(*sensor.state_vec_full_history)

    # # plot light p (noisy, exact and filtered)
    # logger.info("Plotting data")
    # plt.title("Light p")
    # # plt.plot(time_arr_filter, filtered_light_p_homemade, label='Homemade')
    # plt.plot(time_arr_filter, filtered_light_p, label='Filterpy')
    # plt.plot(time_arr, q_p_full_history, label='Exact data')
    # plt.legend()
    # plt.show()

    # # plot light q (noisy, exact and filtered)
    # logger.info("Plotting data")
    # plt.title("Light q")
    # # plt.plot(time_arr_filter, filtered_light_q_homemade, label='Homemade')
    # plt.plot(time_arr_filter, filtered_light_q, label='Filterpy')
    # plt.plot(time_arr, q_q_full_history, label='Exact data')
    # plt.legend()
    # plt.show()

    #plot atoms jy
    logger.info("Plotting data jy")
    plt.title("Atoms jy")
    # plt.plot(time_arr_filter, filtered_atoms_jy_homemade, label='Homemade')
    plt.plot(time_arr_filter, filtered_atoms_jy, label='Filterpy')
    plt.plot(time_arr, j_y_full_history, label='Exact data')
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

    # plot zs
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

    state = PosVelSensorState(initial_vec=np.array([0., 0., 2., 1.]),
                              noise_vec=np.array([GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt),
                                                  GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt),
                                                  GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt),
                                                  GaussianWhiteNoise(0,
                                                                     scalar_strength=0.05,
                                                                     dt=dt)
                                                  ]),
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
