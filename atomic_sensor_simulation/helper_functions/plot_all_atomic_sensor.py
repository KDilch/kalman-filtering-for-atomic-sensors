import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd

from atomic_sensor_simulation.utilities import plot_data, generate_data_arr_for_plotting

plt.style.use('seaborn-darkgrid')

def plot_state_simulation(simulation_state, filename, target_dir='./'):
    # Initialize the figure
    plt.figure(figsize=(14, 9))

    num = 0
    for column in simulation_state.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(simulation_state['time'], simulation_state[column], color='grey', label='Simulation')

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
    # plt.show()
    plt.close()
    return

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
    # plt.show()
    plt.close()
    return

def plot_state_LKF_EKF(LKF_state, EKF_state, simulation_state, filename, target_dir='./'):
    # Initialize the figure
    palette = plt.get_cmap('Set1')
    plt.figure(figsize=(14, 9))

    num = 0
    for column in LKF_state.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(simulation_state['time'][::25], simulation_state[column][::25], linestyle='none', marker='.', color='grey', label='Simulation')
        plt.plot(LKF_state['time'], LKF_state[column], marker='', color=palette(4), linewidth=1.9, alpha=0.9, label='LKF')
        plt.plot(EKF_state['time'], EKF_state[column], marker='', color=palette(2), linewidth=1.9, alpha=0.9, label='EKF')

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
    # plt.show()
    plt.close()
    return

def plot_state_err_cov_LKF(err_cov, err_ss, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in err_cov.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(err_cov['time'], err_cov[column], marker='', color='orange', linewidth=1.9, label=r"LKF estimation error")
        plt.plot(err_ss['time'], err_ss[column], marker='', color='grey', linewidth=1.9, linestyle='--', label='Steady State error')

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
    # plt.show()
    plt.close()
    return

def plot_state_err_LKF(err_cov, err_ss, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in err_cov.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(err_cov['time'], err_cov[column], marker='', color='orange', linewidth=1.9, label=r"LKF estimation error")
        plt.plot(err_ss['time'], err_ss[column], marker='', color='grey', linewidth=1.9, linestyle='--', label='Steady State error')

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
        plt.yscale("log")

    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return

def plot_state_err_LKF_EKF(err_cov_LKF, err_cov_EKF, err_ss, filename, target_dir='./'):
    palette = plt.get_cmap('Set1')
    plt.figure(figsize=(14, 9))

    num = 0
    for column in err_cov_LKF.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 2, num)

        plt.plot(err_cov_LKF['time'], err_cov_LKF[column], marker='', color=palette(4), linewidth=1.9, label=r"LKF estimation error")
        plt.plot(err_cov_LKF['time'], err_cov_EKF[column], marker='', color=palette(2), linewidth=1.9, label=r"EKF estimation error")
        plt.plot(err_ss['time'], err_ss[column], marker='', color='grey', linewidth=1.9, linestyle='--', label='Steady State error')

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
        plt.yscale("log")

    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
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
    # plt.show()
    plt.close()
    return

def plot_zs(zs_real, zs_est, filename, target_dir='./'):
    plt.figure(figsize=(14, 9))

    num = 0
    for column in zs_est.drop('time', axis=1):
        num += 1

        # Find the right spot on the plot
        plt.plot(zs_real['time'][::1], zs_real[column][::1], marker='.', color='grey', linewidth=1.9, label=r"Simulation")
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
    # plt.show()
    plt.close()
    return

def plot_waveform_simulation(waveform_real, filename, target_dir='./'):
    plt.figure(figsize=(14, 5))

    # Find the right spot on the plot
    for column in waveform_real.drop('time', axis=1):
        plt.plot(waveform_real['time'], waveform_real[column], marker='', color='grey', linewidth=1.9, label=r"Simulation")

        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)
    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return

def plot_waveform(waveform_real, waveform_est, waveform_est_err, waveform_ss, filename, target_dir='./'):
    palette = plt.get_cmap('Set1')
    plt.figure(figsize=(14, 5))

    # Find the right spot on the plot
    plt.subplot(1, 2, 1)
    for column in waveform_real.drop('time', axis=1):
        plt.plot(waveform_real['time'], waveform_real[column], marker='', color='grey', linewidth=1.9,
                 label=r"Simulation")
        plt.plot(waveform_est['time'], waveform_est[column], marker='', color=palette(4), linewidth=1.9,
                 label='LKF estimate')

        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column + ' [a.u.]', fontsize=18)
        plt.legend(fontsize=18)

    plt.subplot(1, 2, 2)
    for column in waveform_est_err.drop('time', axis=1):
        plt.plot(waveform_ss['time'], waveform_ss[column], marker='', color='grey', linewidth=1.9,
                 label=r"Steady State error")
        plt.plot(waveform_est_err['time'], waveform_est_err[column], marker='', color=palette(4), linewidth=1.9,
                 label='LKF error')

        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column + ' [a.u.]', fontsize=18)
        plt.yscale('log')
        plt.legend(fontsize=18)

    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return

def plot_waveform_EKF_LKF(waveform_real,
                          waveform_LKF,
                          waveform_EKF,
                          waveform_LKF_err,
                          waveform_EKF_err,
                          waveform_ss,
                          filename,
                          target_dir='./'):
    palette = plt.get_cmap('Set1')
    plt.figure(figsize=(14, 5))

    # Find the right spot on the plot
    plt.subplot(1, 2, 1)
    for column in waveform_real.drop('time', axis=1):
        plt.plot(waveform_real['time'], waveform_real[column], marker='', color='grey', linewidth=1.9, label=r"Simulation")
        plt.plot(waveform_LKF['time'], waveform_LKF[column], marker='', color=palette(4), linewidth=1.9, label='LKF estimate')
        plt.plot(waveform_EKF['time'], waveform_EKF[column], marker='', color=palette(2), linewidth=1.9, label='EKF estimate')


        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column+' [a.u.]', fontsize=18)
        plt.legend(fontsize=18)

    plt.subplot(1, 2, 2)
    for column in waveform_LKF_err.drop('time', axis=1):
        plt.plot(waveform_ss['time'], waveform_ss[column], marker='', color='grey', linewidth=1.9,
                 label=r"Steady State error")
        plt.plot(waveform_LKF_err['time'], waveform_LKF_err[column], marker='', color=palette(4), linewidth=1.9,
                 label='LKF error')
        plt.plot(waveform_EKF_err['time'], waveform_EKF_err[column], marker='', color=palette(2), linewidth=1.9,
                 label='EKF error')

        plt.xlabel('time [a.u.]', fontsize=18)
        plt.ylabel(column + ' [a.u.]', fontsize=18)
        plt.yscale('log')
        plt.legend(fontsize=18)


    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return

def plot_q_est_difference_LKF_EKF(est_diff, est_err_diff, filename, target_dir='./'):
    # Initialize the figure
    plt.figure(figsize=(14, 5))

    # Find the right spot on the plot
    plt.subplot(1, 2, 1)
    plt.plot(est_diff['time'], est_diff['diff'], marker='', color='orange', linewidth=1.9, alpha=0.9)
    plt.xlabel('time [a.u.]', fontsize=18)
    plt.ylabel(r"$|q^{LKF}-q^{EKF}|$" + ' [a.u.]', fontsize=18)
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    plt.subplot(1, 2, 2)
    plt.plot(est_err_diff['time'], est_err_diff['diff'], marker='', color='orange', linewidth=1.9, alpha=0.9)
    plt.xlabel('time [a.u.]', fontsize=18)
    plt.ylabel(r"$|\Delta q^{LKF}-\Delta q^{EKF}|$" + ' [a.u.]', fontsize=18)
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return

def plot_q_est_difference_num_exp(est_diff, est_err_diff, filename, target_dir='./'):
    # Initialize the figure
    plt.figure(figsize=(14, 5))

    # Find the right spot on the plot
    plt.subplot(1, 2, 1)
    plt.plot(est_diff['time'], est_diff['diff'], marker='', color='orange', linewidth=1.9, alpha=0.9)
    plt.xlabel('time [a.u.]', fontsize=18)
    plt.ylabel(r"$|q^{num}-q^{exp}|$" + ' [a.u.]', fontsize=18)

    plt.subplot(1, 2, 2)
    plt.plot(est_err_diff['time'], est_err_diff['diff'], marker='', color='orange', linewidth=1.9, alpha=0.9)
    plt.xlabel('time [a.u.]', fontsize=18)
    plt.ylabel(r"$|\Delta q^{num}-\Delta q^{exp}|$" + ' [a.u.]', fontsize=18)

    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return

def plot_ekf_lin_disc(ekf_lin, ekf_disc, err_ss_q, sim_q, filename, target_dir='./'):
    # Initialize the figure
    palette = plt.get_cmap('Set1')
    plt.figure(figsize=(14, 5))

    # Find the right spot on the plot
    plt.subplot(1, 2, 1)
    plt.plot(ekf_lin['time'], ekf_lin['q'], marker='', color=palette(1), linewidth=1.9, alpha=0.9)
    plt.plot(ekf_disc['time'], ekf_disc['q'], marker='', color=palette(2), linewidth=1.9, alpha=0.9)
    plt.plot(sim_q['time'][::25], sim_q['q'][::25], linestyle='none', marker='.', color='grey',
             label='Simulation')
    plt.xlabel('time [a.u.]', fontsize=18)
    plt.ylabel(r"q" + ' [a.u.]', fontsize=18)

    plt.subplot(1, 2, 2)
    plt.plot(ekf_lin['time'], ekf_lin[r"$\Delta^2$q"], marker='', color=palette(1), linewidth=1.9, alpha=0.9, label="continuous")
    plt.plot(ekf_lin['time'], ekf_disc[r"$\Delta^2$q"], marker='', color=palette(2), linewidth=1.9, alpha=0.9, label="discrete")
    plt.plot(err_ss_q['time'], err_ss_q[r"$\Delta^2$q"], marker='', color='grey', linestyle='--', label='Steady State Error', linewidth=1.9, alpha=0.9)
    plt.xlabel('time [a.u.]', fontsize=18)
    plt.ylabel(r"$\Delta^2$q" + ' [a.u.]', fontsize=18)
    plt.legend(fontsize=18)

    plt.savefig(os.path.join(target_dir, filename))
    # plt.show()
    plt.close()
    return


def plot__all_atomic_sensor(sensor,
                            time_arr_filter,
                            time_arr,
                            lkf_num_history_manager,
                            lkf_expint_approx_history_manager,
                            lkf_exp_approx_history_manager,
                            extended_kf_history_manager,
                            extended_kf_history_manager_lin,
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
    target = "./Simulation_plots/gp_%r_wp_%r" % (int(config.coupling['g_p']), int(config.coupling['omega_p']))
    if not os.path.exists(target):
        os.makedirs(target)
    # Get history data from sensor state class and separate into blocks using "zip".
    j_y_full_history, j_z_full_history, q_q_full_history, q_p_full_history = zip(*sensor.state_vec_full_history)

    #PLOT STATE VECTOR COMPONENTS AND LKF
    simulation_state = pd.DataFrame({'time': time_arr, r"J$_y$": j_y_full_history, r"J$_z$": j_z_full_history, 'q': q_q_full_history, 'p': q_p_full_history})
    LKF_state = pd.DataFrame({'time': time_arr_filter, r"J$_y$": lkf_num_history_manager.jys, r"J$_z$": lkf_num_history_manager.jzs,
                       'q': lkf_num_history_manager.qs, 'p': lkf_num_history_manager.ps})
    plot_state_LKF(LKF_state, simulation_state, filename='plt_state_gp_%r_wp_%r.png'%(config.coupling['g_p'], config.coupling['omega_p']), target_dir=target)

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
                       filename='plt_state_err_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']),
                       target_dir=target)

    # #PLOT REAL ESTIMATION ERR SQ(np.dot((x-x_est), (x-x_est).T))
    # err_LKF = pd.DataFrame(
    #     {'time': time_arr_filter,
    #      r"$\Delta^2$J$_y$": error_jy_LKF,
    #      r"$\Delta^2$J$_z$": error_jz_LKF,
    #      r"$\Delta^2$q": error_q_LKF,
    #      r"$\Delta^2$p": error_p_LKF})
    # # err_EKF = pd.DataFrame(
    # #     {'time': time_arr_filter,
    # #      r"$\Delta^2$J$_y$": error_jy_EKF,
    # #      r"$\Delta^2$J$_z$": error_jz_EKF,
    # #      r"$\Delta^2$q": error_q_EKF,
    # #      r"$\Delta^2$p": error_p_EKF})
    # plot_state_real_err_LKF(err_LKF,
    #                         err_cov,
    #                         filename='plt_state_real_err_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']))

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
    plot_zs(zs_real, zs_est_LKF, 'plt_zs_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']), target_dir=target)

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
    plot_waveform(waveform_real, waveform_LKF, waveform_err_cov, waveform_err_ss, 'plt_waveform_gp_%r_wp_%r.png' % (config.coupling['g_p'], config.coupling['omega_p']), target_dir=target)

    #PLOT DIFFERENCE LKF-EKF ESTIMATES
    diff_LKF_EKF = pd.DataFrame(
        {
            'time': time_arr_filter,
            'diff': np.abs(lkf_num_history_manager.qs-extended_kf_history_manager.qs)
        })

    diff_LKF_EKF_err = pd.DataFrame(
        {
            'time': time_arr_filter,
            'diff': np.abs(np.sqrt(lkf_num_history_manager.qs_err_post) - np.sqrt(extended_kf_history_manager.qs_err_post))
        })
    plot_q_est_difference_LKF_EKF(diff_LKF_EKF, diff_LKF_EKF_err, 'plt_q_LKF_EKF_difference_gp_%r_wp_%r.png' % (
    config.coupling['g_p'], config.coupling['omega_p']), target_dir=target)

    # PLOT DIFFERENCE LKF-num_exp ESTIMATES
    diff_LKF_num_exp = pd.DataFrame(
        {
            'time': time_arr_filter,
            'diff': np.abs(lkf_num_history_manager.qs - lkf_exp_approx_history_manager.qs)
        })
    diff_LKF_EKF_num_exp_err = pd.DataFrame(
        {
            'time': time_arr_filter,
            'diff': np.abs(np.sqrt(lkf_num_history_manager.qs_err_post) - np.sqrt(lkf_exp_approx_history_manager.qs_err_post))
        })
    plot_q_est_difference_num_exp(diff_LKF_num_exp, diff_LKF_EKF_num_exp_err, 'plt_LKF_num_exp_difference_gp_%r_wp_%r.png' % (
    config.coupling['g_p'], config.coupling['omega_p']), target_dir=target)


    #PLOT EKF LIN AND DISC
    ekf_lin = pd.DataFrame(
        {
            'time': time_arr_filter,
            'q': extended_kf_history_manager_lin.qs,
            r"$\Delta^2$q": extended_kf_history_manager_lin.qs_err_post
        })
    ekf_disc = pd.DataFrame(
        {
            'time': time_arr_filter,
            'q': extended_kf_history_manager.qs,
            r"$\Delta^2$q": extended_kf_history_manager.qs_err_post
        })
    err_ss_q = pd.DataFrame(
        {'time': time_arr_filter,
         r"$\Delta^2$q": steady_state_history_manager.steady_posts_q})
    sim_q = pd.DataFrame({'time': time_arr, 'q': q_q_full_history})

    plot_ekf_lin_disc(ekf_lin, ekf_disc, err_ss_q, sim_q,
                                  'plt_EKF_lin_disc_gp_%r_wp_%r.png' % (
                                      config.coupling['g_p'], config.coupling['omega_p']), target_dir=target)

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