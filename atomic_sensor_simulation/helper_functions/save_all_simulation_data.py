#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from utilities import generate_data_arr_for_saving


def save_data(sensor,
              time_arr_filter,
              time_arr,
              lkf_num_history_manager,
              extended_kf_history_manager,
              unscented_kf_history_manager,
              steady_state_history_manager,
              zs_filter_freq,
              args,
              file_basename):
    # PLOTS=========================================================
    logger = logging.getLogger(__name__)
    logger.info('Preparing data for saving.')
    # Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_q_full_history, q_p_full_history = zip(*sensor.state_vec_full_history)

    labels_filter = ['time_arr_filter',
                     'jy_lin', 'jy_ext', 'jy_unsc', 'jy_err_lin', 'jy_err_ext', 'jy_err_unsc', 'jy_steady_err',
                     'z',
                     'jz_lin', 'jz_ext', 'jz_unsc', 'jz_err_lin', 'jz_err_ext', 'jz_err_unsc', 'jz_steady_err',
                     'q_lin', 'q_ext', 'q_unsc', 'q_err_lin', 'q_err_ext', 'q_err_unsc', 'q_steady_err',
                     'p_lin', 'p_ext', 'p_unsc', 'p_err_lin', 'p_err_ext', 'p_err_unsc', 'p_steady_err']
    labels_simulation = ['time_arr_simulation', 'jy_real', 'jz_real', 'q_real', 'p_real']

    if np.any([args[0].lkf_num, args[0].lkf_exp, args[0].lkf_expint, args[0].ekf, args[0].ukf]):
        logger.info("Saving simulation data to csv file")
        all_data_filter = generate_data_arr_for_saving(np.array([time_arr_filter,
                                                                lkf_num_history_manager.jys,
                                                                extended_kf_history_manager.jys,
                                                                unscented_kf_history_manager.jys,
                                                                 lkf_num_history_manager.jys_err_post,
                                                                 extended_kf_history_manager.jys_err_post,
                                                                 unscented_kf_history_manager.jys_err_post,
                                                                 steady_state_history_manager.steady_posts_jy,
                                                                 zs_filter_freq,
                                                                 lkf_num_history_manager.jzs,
                                                                 extended_kf_history_manager.jzs,
                                                                 unscented_kf_history_manager.jzs,
                                                                 lkf_num_history_manager.jzs_err_post,
                                                                 extended_kf_history_manager.jzs_err_post,
                                                                 unscented_kf_history_manager.jzs_err_post,
                                                                 steady_state_history_manager.steady_posts_jz,
                                                                 lkf_num_history_manager.qs,
                                                                 extended_kf_history_manager.qs,
                                                                 unscented_kf_history_manager.qs,
                                                                 lkf_num_history_manager.qs_err_post,
                                                                 extended_kf_history_manager.qs_err_post,
                                                                 unscented_kf_history_manager.qs_err_post,
                                                                 steady_state_history_manager.steady_posts_q,
                                                                 lkf_num_history_manager.ps,
                                                                 extended_kf_history_manager.ps,
                                                                 unscented_kf_history_manager.ps,
                                                                 lkf_num_history_manager.ps_err_post,
                                                                 extended_kf_history_manager.ps_err_post,
                                                                 unscented_kf_history_manager.ps_err_post,
                                                                 steady_state_history_manager.steady_posts_p
                                                                 ]),
                                                                    labels=labels_filter,
                                                                    bools=[True,
                                                                        args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True,
                                                                           True,
                                                                           args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True,
                                                                           args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True,
                                                                           args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           args[0].lkf_num,
                                                                           args[0].ekf,
                                                                           args[0].ukf,
                                                                           True])
        all_data_simulation = generate_data_arr_for_saving(np.array([time_arr,
                                                                     j_y_full_history,
                                                                     j_z_full_history,
                                                                     q_q_full_history,
                                                                     q_p_full_history]),
                                                       labels=['time',
                                                               'Jy',
                                                               'Jz',
                                                               'q',
                                                               'p'],
                                                       bools=[True,
                                                              True,
                                                              True,
                                                              True,
                                                              True])
        all_data_filter.to_csv(file_basename + '_kf_gp_150_wp_0.csv', sep='\t', na_rep='Unknown')
        all_data_simulation.to_csv(file_basename + '_sim_gp_150_wp_0.csv', sep='\t', na_rep='Unknown')
    return