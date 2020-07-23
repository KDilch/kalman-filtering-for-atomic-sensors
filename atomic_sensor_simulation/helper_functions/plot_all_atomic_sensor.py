import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd

from atomic_sensor_simulation.utilities import plot_data, generate_data_arr_for_plotting

plt.style.use('seaborn-darkgrid')

def plot_state_LKF(filter_state, simulation_state, filename, target_dir='./'):
    # Initialize the figure
    plt.figure(figsize=(14, 9))

    num = 0
    for column in filter_state.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(simulation_state['time'][::25], simulation_state[column][::25], linestyle='none', marker='.', color='grey', label='Simulation')
        plt.plot(filter_state['time'], filter_state[column], marker='', color='orange', linewidth=1.9, alpha=0.9, label='LKF')

        if num in [4, 3]:
            plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)

        if num == 4:
            plt.legend(fontsize=18)

        # Not ticks everywhere
        if num in range(7):
            plt.tick_params(labelbottom='off')
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    return

def plot_state_err_LKF(err_cov, err_ss, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in err_cov.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(err_cov['time'], err_cov[column], marker='', color='orange', linewidth=1.9, label=r"LKF $\{\Sigma_{k|k}\}_{i}^{i}$")
        plt.plot(err_ss['time'], err_ss[column], marker='', color='grey', linewidth=1.9, linestyle='--', label='Steady State Error')

        if num in [4, 3]:
            plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)

        if num == 4:
            plt.legend(fontsize=18)

        # Not ticks everywhere
        if num in range(7):
            plt.tick_params(labelbottom='off')
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    return

def plot_state_real_err_LKF(err_real, err_ss, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in err_real.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(err_real['time'], err_real[column], marker='', color='orange', linewidth=1.9, label=r"LKF $(x-\tilde{x})(x-\tilde{x})^T$")
        plt.plot(err_ss['time'], err_ss[column], marker='', color='grey', linewidth=1.9, linestyle='--', label='Steady State Error')

        if num in [4, 3]:
            plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)

        if num == 4:
            plt.legend(fontsize=18)

        # Not ticks everywhere
        if num in range(7):
            plt.tick_params(labelbottom='off')
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    return

def plot_zs(zs_real, zs_est, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in zs_est.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.plot(zs_real['time'], zs_real[column], marker='', color='grey', linewidth=1.9, linestyle='--', label=r"Simulation")
        plt.plot(zs_est['time'], zs_est[column], marker='', color='orange', linewidth=1.9, label='LKF estimate')


        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)
        plt.legend(fontsize=18)

        # Not ticks everywhere
        if num in range(7):
            plt.tick_params(labelbottom='off')
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    return

def plot_waveform(waveform_real, waveform_est, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in waveform_real.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.plot(waveform_real['time'], waveform_real[column], marker='', color='grey', linewidth=1.9, label=r"Simulation")
        plt.plot(waveform_est['time'], waveform_est[column], marker='', color='orange', linewidth=1.9, label='LKF estimate')


        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)
        plt.legend(fontsize=18)

        # Not ticks everywhere
        if num in range(7):
            plt.tick_params(labelbottom='off')
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    return

def plot_waveform_err(waveform_real_err, waveform_ss_err, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in waveform_real_err.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.plot(waveform_ss_err['time'], waveform_ss_err[column], marker='', color='grey', linewidth=1.9, label=r"Steady State error")
        plt.plot(waveform_real_err['time'], waveform_real_err[column], marker='', color='orange', linewidth=1.9, label='LKF error')


        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)
        plt.legend(fontsize=18)

        # plt.xlim((0., 5.))
        # plt.ylim((0., 500))

        # Not ticks everywhere
        if num in range(7):
            plt.tick_params(labelbottom='off')
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return

def plot_EKF_LKF_DIFFERENCE(diff, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in diff.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        # plt.plot(waveform_ss_err['time'], waveform_ss_err[column], marker='', color='grey', linewidth=1.9, label=r"Steady State error")
        plt.plot(diff['time'], diff[column], marker='', color='orange', linewidth=1.9)


        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)

        # Not ticks everywhere
        if num in range(7):
            plt.tick_params(labelbottom='off')
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    return

# def plot_histogram(waveform_err_ratio, filename, target_dir='./'):
#     import seaborn as sns
#     # Control the number of bins
#     sns.distplot(waveform_err_ratio['ratios'])
#     plt.show()
#     plt.savefig(os.path.join(target_dir, filename))
#     plt.close()
#     return


def plot__all_atomic_sensor(sensor,
                            time_arr_filter,
                            time_arr,
                            lkf_num_history_manager,
                            lkf_expint_approx_history_manager,
                            lkf_exp_approx_history_manager,
                            extended_kf_history_manager,
                            unscented_kf_history_manager,
                            steady_state_history_manager,
                            zs,
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
                            zs_sigma,
                            args,
                            config):
    # PLOTS=========================================================
    logger = logging.getLogger(__name__)
    logger.info('Plotting data.')
    # Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_q_full_history, q_p_full_history = zip(*sensor.state_vec_full_history)

    #PLOT STATE VECTOR COMPONENTS AND LKF
    simulation_state = pd.DataFrame({'time': time_arr, r"J$_y$": j_y_full_history, r"J$_z$": j_z_full_history, 'q': q_q_full_history, 'p': q_p_full_history})
    LKF_state = pd.DataFrame({'time': time_arr_filter, r"J$_y$": lkf_num_history_manager.jys, r"J$_z$": lkf_num_history_manager.jzs,
                       'q': lkf_num_history_manager.qs, 'p': lkf_num_history_manager.ps})
    plot_state_LKF(LKF_state, simulation_state, filename='plt_state_gp_%r_wp_%r.png'%(config.coupling['g_p'], config.coupling['omega_p']))

    # PLOT STATE VECTOR ESTIMATION ERR
    err_cov = pd.DataFrame(
        {'time': time_arr_filter,
         r"$\Delta^2$J$_y$": lkf_num_history_manager.jys_err_post,
         r"$\Delta^2$J$_z$": lkf_num_history_manager.jzs_err_post,
         r"$\Delta^2$q": lkf_num_history_manager.qs_err_post,
         r"$\Delta^2$p": lkf_num_history_manager.ps_err_post})
    err_ss = pd.DataFrame(
        {'time': time_arr_filter,
         r"$\Delta^2$J$_y$": steady_state_history_manager.steady_posts_jy,
         r"$\Delta^2$J$_z$": steady_state_history_manager.steady_posts_jz,
         r"$\Delta^2$q": steady_state_history_manager.steady_posts_q,
         r"$\Delta^2$p": steady_state_history_manager.steady_posts_p})
    plot_state_err_LKF(err_cov,
                       err_ss,
                       filename='plt_state_err_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))

    #PLOT REAL ESTIMATION ERR SQ(np.dot((x-x_est), (x-x_est).T))
    err_LKF = pd.DataFrame(
        {'time': time_arr_filter,
         r"$\Delta^2$J$_y$": error_jy_LKF,
         r"$\Delta^2$J$_z$": error_jz_LKF,
         r"$\Delta^2$q": error_q_LKF,
         r"$\Delta^2$p": error_p_LKF})
    # err_EKF = pd.DataFrame(
    #     {'time': time_arr_filter,
    #      r"$\Delta^2$J$_y$": error_jy_EKF,
    #      r"$\Delta^2$J$_z$": error_jz_EKF,
    #      r"$\Delta^2$q": error_q_EKF,
    #      r"$\Delta^2$p": error_p_EKF})
    plot_state_real_err_LKF(err_LKF,
                            err_ss,
                            filename='plt_state_real_err_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))

    #PLOT zs/sigma
    zs_real = pd.DataFrame(
        {
            'time': time_arr_filter,
            "$z_k/\sigma_D$": zs_sigma
        }
    )
    zs_est_LKF = pd.DataFrame(
        {
            'time': time_arr_filter,
            "$z_k/\sigma_D$": lkf_num_history_manager.zs_est
        }
    )
    plot_zs(zs_real, zs_est_LKF, 'plt_zs_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))

    #PLOT WAVEFORM
    waveform_real = pd.DataFrame(
        {
            'time': time_arr,
            r"$\varepsilon$": sensor.waveform_history
        }
    )
    waveform_LKF = pd.DataFrame(
        {
            'time': time_arr_filter,
            r"$\varepsilon$": lkf_num_history_manager.waveform_est
        }
    )
    plot_waveform(waveform_real, waveform_LKF, 'plt_waveform_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))

    #PLOT WAVEFORM ERR
    waveform_err_real = pd.DataFrame(
        {
            'time': time_arr_filter,
            r"$\Delta^2 \varepsilon$": error_waveform_LKF
        }
    )
    waveform_err_cov = pd.DataFrame(
        {
            'time': time_arr_filter,
            r"$\Delta^2 \varepsilon$": lkf_num_history_manager.waveform_est_err
        }
    )
    waveform_err_ss = pd.DataFrame(
        {
            'time': time_arr_filter,
            r"$\Delta^2 \varepsilon$": steady_state_history_manager.steady_waveform_err
        }
    )
    plot_waveform_err(waveform_err_real, waveform_err_ss, 'plt_waveform_err_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))

    # #PLOT WAVEFORM HISTOGRAM
    # waveform_histogram = pd.DataFrame(
    #     {
    #         'ratios': error_waveform_LKF
    #     }
    # )
    # plot_histogram(waveform_histogram, 'plt_waveform_histogram_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))
    #

    #PLOT DIFFERENCE LKF-EKF ESTIMATES
    diff_LKF_EKF = pd.DataFrame(
        {
            'time': time_arr_filter,
            r"$|q^{LKF}-q^{EKF}|$": np.abs(lkf_num_history_manager.qs-extended_kf_history_manager.qs)
        })
    plot_EKF_LKF_DIFFERENCE(diff_LKF_EKF, 'plt_q_est_difference_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))

    # PLOT DIFFERENCE LKF-EKF ESTIMATES (COV)
    diff_LKF_EKF = pd.DataFrame(
        {
            'time': time_arr_filter,
            r"$|\Delta q^{LKF}-\Delta q^{EKF}|$": np.abs(np.sqrt(lkf_num_history_manager.qs_err_post) - np.sqrt(extended_kf_history_manager.qs_err_post))
        })
    plot_EKF_LKF_DIFFERENCE(diff_LKF_EKF, 'plt_q_err_est_difference_gp_%r_wp_%r.png' % (
    config.coupling['g_p'], config.coupling['omega_p']))

    # labels = ['Linear kf num', 'Linear kf expint', 'Linear kf exp', 'Extended kf', 'Unscented kf', 'Exact data']
    # labels_err = ['Linear kf num err', 'Linear kf expint err', 'Linear kf exp err', 'Extended kf err',
    #               'Unscented kf err', 'Steady state']
    # # plot atoms jy
    # if np.any([args[0].lkf_num, args[0].lkf_exp, args[0].lkf_expint, args[0].ekf, args[0].ukf]):
    #     logger.info("Plotting data jy")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr]),
    #                                                                 np.array([lkf_num_history_manager.jys,
    #                                                                           lkf_expint_approx_history_manager.jys,
    #                                                                           lkf_exp_approx_history_manager.jys,
    #                                                                           extended_kf_history_manager.jys,
    #                                                                           unscented_kf_history_manager.jys,
    #                                                                           j_y_full_history]),
    #                                                                 labels=labels,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Atoms jy", is_show=True, is_legend=True)
    #
    #     # plot error for atoms jy
    #     logger.info("Plotting error jy")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter]),
    #                                                                 np.array([lkf_num_history_manager.jys_err_post,
    #                                                                           lkf_expint_approx_history_manager.jys_err_post,
    #                                                                           lkf_exp_approx_history_manager.jys_err_post,
    #                                                                           extended_kf_history_manager.jys_err_post,
    #                                                                           unscented_kf_history_manager.jys_err_post,
    #                                                                           steady_state_history_manager.steady_posts_jy]),
    #                                                                 labels=labels_err,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error jy", is_show=True, is_legend=True)
    #
    #     # plot atoms jz
    #     logger.info("Plotting data jz")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr]),
    #                                                                 np.array([lkf_num_history_manager.jzs,
    #                                                                           lkf_expint_approx_history_manager.jzs,
    #                                                                           lkf_exp_approx_history_manager.jzs,
    #                                                                           extended_kf_history_manager.jzs,
    #                                                                           unscented_kf_history_manager.jzs,
    #                                                                           j_z_full_history]),
    #                                                                 labels=labels,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Atoms jz", is_show=True, is_legend=True)
    #
    #     # plot error for atoms jz
    #     logger.info("Plotting error jz")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter]),
    #                                                                 np.array([lkf_num_history_manager.jzs_err_post,
    #                                                                           lkf_expint_approx_history_manager.jzs_err_post,
    #                                                                           lkf_exp_approx_history_manager.jzs_err_post,
    #                                                                           extended_kf_history_manager.jzs_err_post,
    #                                                                           unscented_kf_history_manager.jzs_err_post,
    #                                                                           steady_state_history_manager.steady_posts_jz]),
    #                                                                 labels=labels_err,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error jz", is_show=True, is_legend=True)
    #
    #     # plot light q (noisy, exact and filtered)
    #     logger.info("Plotting data q")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr]),
    #                                                                 np.array([lkf_num_history_manager.qs,
    #                                                                           lkf_expint_approx_history_manager.qs,
    #                                                                           lkf_exp_approx_history_manager.qs,
    #                                                                           extended_kf_history_manager.qs,
    #                                                                           unscented_kf_history_manager.qs,
    #                                                                           q_q_full_history]),
    #                                                                 labels=labels,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="q", is_show=True, is_legend=True)
    #
    #     # plot error for light q
    #     logger.info("Plotting error q")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter]),
    #                                                                 np.array([lkf_num_history_manager.qs_err_post,
    #                                                                           lkf_expint_approx_history_manager.qs_err_post,
    #                                                                           lkf_exp_approx_history_manager.qs_err_post,
    #                                                                           extended_kf_history_manager.qs_err_post,
    #                                                                           unscented_kf_history_manager.qs_err_post,
    #                                                                           steady_state_history_manager.steady_posts_q]),
    #                                                                 labels=labels_err,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error q", is_show=True, is_legend=True)
    #
    #     # plot light p (noisy, exact and filtered)
    #     logger.info("Plotting data p")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr]),
    #                                                                 np.array([lkf_num_history_manager.ps,
    #                                                                           lkf_expint_approx_history_manager.ps,
    #                                                                           lkf_exp_approx_history_manager.ps,
    #                                                                           extended_kf_history_manager.ps,
    #                                                                           unscented_kf_history_manager.ps,
    #                                                                           q_p_full_history]),
    #                                                                 labels=labels,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Atoms p", is_show=True, is_legend=True)
    #
    #     # plot error for atoms p
    #     logger.info("Plotting error p")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter,
    #                                                                           time_arr_filter]),
    #                                                                 np.array([lkf_num_history_manager.ps_err_post,
    #                                                                           lkf_expint_approx_history_manager.ps_err_post,
    #                                                                           lkf_exp_approx_history_manager.ps_err_post,
    #                                                                           extended_kf_history_manager.ps_err_post,
    #                                                                           unscented_kf_history_manager.ps_err_post,
    #                                                                           steady_state_history_manager.steady_posts_p]),
    #                                                                 labels=labels_err,
    #                                                                 bools=[args[0].lkf_num,
    #                                                                        args[0].lkf_exp,
    #                                                                        args[0].lkf_expint,
    #                                                                        args[0].ekf,
    #                                                                        args[0].ukf,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Error p", is_show=True, is_legend=True)