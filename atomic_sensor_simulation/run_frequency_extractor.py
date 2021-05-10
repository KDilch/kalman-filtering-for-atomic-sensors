#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np

from atomic_sensor_simulation.utilities import import_config_from_path
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.dynamical_model.freq_state import FrequencySensorState
from atomic_sensor_simulation.sensor.frequency_sensor import FrequencySensor
from filter_model.FrequencyExtractor.extended_kf import Extended_KF
from atomic_sensor_simulation.utilities import calculate_error, plot_data, \
    generate_data_arr_for_plotting
from history_manager.frequency_extractor_history_manager import Filter_History_Manager


def run__frequency_extractor(*args):

    # Logger for storing errors and logs in seperate file, creates separate folder
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

    logger.info('Setting initial dynamical_model vec to  [%r, %r, %r].' %
                (str(config.simulation['x1']),
                 str(config.simulation['x2']),
                 str(config.simulation['x3'])
                 )
                )

    num_iter_sensor = config.simulation['num_iter_sensor']
    num_iter_filter = np.int(np.floor_divide(num_iter_sensor * config.simulation['dt_sensor'],
                                             config.filter['dt_filter']))

    every_nth_z = np.int(np.floor_divide(num_iter_sensor, num_iter_filter))

    Q = np.array([[config.noise_and_measurement['Qx1'], 0., 0.],
                  [0., config.noise_and_measurement['Qx2'], 0.],
                  [0., 0., config.noise_and_measurement['Qx3']]])
    H = np.array([[1., 0., 0.]])
    R = np.array([[config.noise_and_measurement['R']]])

    logger.info('Setting Q, H and R to Q = %r, H = %r, R = %r' %
                (str(Q),
                 str(H),
                 str(R)
                 )
                )

    time_arr = np.arange(0, num_iter_sensor * config.simulation['dt_sensor'], config.simulation['dt_sensor'])
    time_arr_filter = np.arange(0, num_iter_filter * config.filter['dt_filter'], config.filter['dt_filter'])

    # SIMULATING DYNAMICS=====================================================

    state = FrequencySensorState(initial_vec=np.array([config.simulation['x1'],
                                                       config.simulation['x2'],
                                                       config.simulation['x3']]),
                                 noise_vec=GaussianWhiteNoise(mean=[0., 0., 0.],
                                                              cov=Q,
                                                              dt=config.simulation['dt_sensor']),
                                 initial_time=0,
                                 time_arr=time_arr,
                                 dt=config.simulation['dt_sensor'])

    sensor = FrequencySensor(state,
                             sensor_noise=GaussianWhiteNoise(mean=0.,
                                                             cov=R / config.simulation['dt_sensor'],
                                                             dt=config.simulation['dt_sensor']),
                             H=H,
                             dt=config.simulation['dt_sensor'])

    zs = np.array([np.array((sensor.read(_))) for _ in time_arr])  # noisy measurement
    zs_filter_freq = zs[::every_nth_z]
    # x_filter_freq = sensor.state_vec_full_history[::every_nth_z]

    # KALMAN FILTER====================================================
    # unscented_kf_model = Unscented_KF(fx=compute_fx_at_time_t(0),
    #                                   Q=linear_kf_model.Q_delta,
    #                                   hx=hx,
    #                                   R=R / config.filter['dt_filter'],
    #                                   Gamma=dynamical_model.Gamma_control_evolution_matrix,
    #                                   u=dynamical_model.u_control_vec,
    #                                   z0=[zs[0]],
    #                                   dt=config.filter['dt_filter'],
    #                                   x0=linear_kf_model.x0,
    #                                   P0=linear_kf_model.P0)

    extended_kf_model = Extended_KF(Q=Q,
                                    H=H,
                                    R_delta=R / config.filter['dt_filter'],
                                    Gamma=state.Gamma_control_evolution_matrix,
                                    u=state.u_control_vec,
                                    z0=[zs[0]],
                                    dt=config.filter['dt_filter'],
                                    num_terms=3,
                                    time_arr=time_arr_filter
                                    )

    # RUN FILTERPY KALMAN FILTER
    logger.info("Initializing linear_kf_filterpy Kalman Filter")

    # logger.info("Initializing unscented_kf_filterpy Unscented Filter")
    # unscented_kf_filterpy = unscented_kf_model.initialize_filterpy()
    # unscented_kf_history_manager = Filter_History_Manager(unscented_kf_filterpy, num_iter_filter)

    logger.info("Initializing extended_kf_filterpy Unscented Filter")
    extended_kf_filterpy = extended_kf_model.initialize_filterpy()
    extended_kf_history_manager = Filter_History_Manager(extended_kf_filterpy, num_iter_filter)

    for index, time in enumerate(time_arr_filter):
        z = zs_filter_freq[index]
        # unscented_kf_filterpy.predict(fx=compute_fx_at_time_t(time))
        # unscented_kf_filterpy.update(z)
        # unscented_kf_history_manager.add_entry(index)

        extended_kf_filterpy.predict()
        print(H, "H")
        extended_kf_filterpy.update(z, HJacobian=lambda x: H, Hx=lambda x: np.dot(H, x))
        extended_kf_history_manager.add_entry(index)

    # PLOTS=========================================================
    # Get history data from sensor dynamical_model class and separate into blocks using "zip".
    x1_full_history, x2_full_history, x3_full_history = zip(*sensor.state_vec_full_history)

    labels = ['Extended kf', 'Exact data']
    labels_err = ['Extended kf err', 'Steady dynamical_model']

    if np.any([True]):
        logger.info("Plotting data x1")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
                                                 time_arr_filter,
                                                 time_arr]),
                                       np.array([extended_kf_history_manager.x1s,
                                                 x1_full_history]),
                                       labels=labels,
                                       bools=[True,
                                             True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="x1", is_show=True, is_legend=True)

        # plot error for atoms jy
        logger.info("Plotting error x1")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
                                                                              time_arr_filter]),
                                                                    np.array([extended_kf_history_manager.x1s_err_post,
                                                                              ]),
                                                                    labels=labels_err,
                                                                    bools=[True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error x1", is_show=True, is_legend=True)

        # plot atoms jz
        logger.info("Plotting data x2")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
                                                                              time_arr_filter,
                                                                              time_arr]),
                                                                    np.array([extended_kf_history_manager.x2s,
                                                                              x2_full_history]),
                                                                    labels=labels,
                                                                    bools=[True,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="x2", is_show=True, is_legend=True)

        # plot error for atoms jz
        logger.info("Plotting error x2")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
                                                                              time_arr_filter]),
                                                                    np.array([extended_kf_history_manager.x2s_err_post,
                                                                             ]),
                                                                    labels=labels_err,
                                                                    bools=[True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error x2", is_show=True, is_legend=True)

        logger.info("Plotting data x3")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
                                                                              time_arr_filter,
                                                                              time_arr]),
                                                                    np.array([
                                                                              extended_kf_history_manager.x3s,
                                                                              x3_full_history]),
                                                                    labels=labels,
                                                                    bools=[True,
                                                                           True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="x3", is_show=True, is_legend=True)

        # plot error for light q
        logger.info("Plotting error q")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([

                                                                              time_arr_filter]),
                                                                    np.array([
                                                                              extended_kf_history_manager.x3s_err_post,
                                                                             ]),
                                                                    labels=labels_err,
                                                                    bools=[True])
        plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error x3", is_show=True, is_legend=True)
