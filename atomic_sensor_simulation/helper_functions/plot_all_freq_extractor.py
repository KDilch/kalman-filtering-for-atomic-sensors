import numpy as np
import logging

from atomic_sensor_simulation.utilities import  plot_data, generate_data_arr_for_plotting

def plot__all_atomic_sensor(sensor,
                            time_arr_filter,
                            time_arr,
                            lkf_num_history_manager,
                            lkf_expint_approx_history_manager,
                            lkf_exp_approx_history_manager,
                            extended_kf_history_manager,
                            unscented_kf_history_manager,
                            steady_state_history_manager,
                            args):
    # PLOTS=========================================================
    logger = logging.getLogger(__name__)
    logger.info('Plotting data.')
    # Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_q_full_history, q_p_full_history = zip(*sensor.state_vec_full_history)

    labels = ['Linear kf num', 'Linear kf expint', 'Linear kf exp', 'Extended kf', 'Unscented kf', 'Exact data']
    labels_err = ['Linear kf num err', 'Linear kf expint err', 'Linear kf exp err', 'Extended kf err',
                  'Unscented kf err', 'Steady state']

    # plot atoms jy
    if np.any([args[0].lkf_num, args[0].lkf_exp, args[0].lkf_expint, args[0].ekf, args[0].ukf]):
        logger.info("Plotting data jy")
        xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr]),
                                                                    np.array([lkf_num_history_manager.x1s,
                                                                              lkf_expint_approx_history_manager.x1s,
                                                                              lkf_exp_approx_history_manager.x1s,
                                                                              extended_kf_history_manager.x1s,
                                                                              unscented_kf_history_manager.x1s,
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
                                                                    np.array([lkf_num_history_manager.x1s_err_post,
                                                                              lkf_expint_approx_history_manager.x1s_err_post,
                                                                              lkf_exp_approx_history_manager.x1s_err_post,
                                                                              extended_kf_history_manager.x1s_err_post,
                                                                              unscented_kf_history_manager.x1s_err_post,
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
                                                                    np.array([lkf_num_history_manager.x2s,
                                                                              lkf_expint_approx_history_manager.x2s,
                                                                              lkf_exp_approx_history_manager.x2s,
                                                                              extended_kf_history_manager.x2s,
                                                                              unscented_kf_history_manager.x2s,
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
                                                                    np.array([lkf_num_history_manager.x2s_err_post,
                                                                              lkf_expint_approx_history_manager.x2s_err_post,
                                                                              lkf_exp_approx_history_manager.x2s_err_post,
                                                                              extended_kf_history_manager.x2s_err_post,
                                                                              unscented_kf_history_manager.x2s_err_post,
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
                                                                    np.array([lkf_num_history_manager.x3s_err_post,
                                                                              lkf_expint_approx_history_manager.x3s_err_post,
                                                                              lkf_exp_approx_history_manager.x3s_err_post,
                                                                              extended_kf_history_manager.x3s_err_post,
                                                                              unscented_kf_history_manager.x3s_err_post,
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
                            time_arr_filter,
                                                                              time_arr_filter,
                                                                              time_arr_filter,                                                                              extended_kf_history_manager.ps,
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