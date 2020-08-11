# -*- coding: utf-8 -*-
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from atomic_sensor_simulation.helper_functions.plot_all_atomic_sensor import *
import os

plt.style.use('seaborn-darkgrid')

labels_filter = ['time_arr_filter',
                 'jy_lin', 'jy_lin_approx', 'jy_ext', 'jy_ext_lin', 'jy_unsc', 'jy_err_lin_cov',
                 'jy_err_lin_approx_cov', 'jy_err_ext_cov', 'jy_err_ext_lin_cov', 'jy_err_unsc_cov', 'jy_err_lin',
                 'jy_err_ext', 'jy_steady_err',
                 'z',
                 'jz_lin', 'jz_lin_approx', 'jz_ext', 'jz_ext_lin', 'jz_unsc', 'jz_err_lin_cov',
                 'jz_err_lin_approx_cov', 'jz_err_ext_cov', 'jz_err_unsc_cov', 'jz_err_ext_lin_cov', 'jz_err_lin',
                 'jz_err_ext', 'jz_steady_err',
                 'q_lin', 'q_lin_approx', 'q_ext', 'q_ext_lin', 'q_unsc', 'q_err_lin_cov', 'q_err_lin_approx_cov',
                 'q_err_ext_cov', 'q_err_ext_lin_cov', 'q_err_unsc_cov', 'q_err_lin',
                 'q_err_ext', 'q_steady_err',
                 'p_lin', 'p_lin_approx', 'p_ext', 'p_ext_lin', 'p_unsc', 'p_err_lin_cov', 'p_err_lin_approx_cov',
                 'p_err_ext_cov', 'p_err_ext_lin_cov', 'p_err_unsc_cov', 'p_err_lin', 'p_err_ext', 'p_steady_err',
                 'waveform_est_LKF', 'waveform_est_EKF', 'waveform_est_LKF_err', 'waveform_est_EKF_err',
                 'waveform_real_LKF_err', 'waveform_real_EKF_err', 'waveform_ss_error', 'z/sigma']
labels=['time',
'Jy',
'Jz',
'q',
'p',
'waveform']


def plot__all_atomic_sensor_from_files(gp, w_p, target_dir='./'):
    data_sim = pd.read_csv("./Simulation_data/data_sim_gp_%r_wp_%r.csv" % (gp, w_p), sep='\t')
    data_kf = pd.read_csv("./Simulation_data/data_kf_gp_%r_wp_%r.csv" % (gp, w_p), sep='\t')
    logger = logging.getLogger(__name__)
    logger.info('Plotting data from data files.')
    # PLOT STATE VECTOR COMPONENTS AND LKF
    simulation_state = pd.DataFrame(
        {'time': data_sim['time'], r"J$_y$": data_sim['Jy'], r"J$_z$": data_sim['Jz'], 'q': data_sim['q'],
         'p': data_sim['p']})
    LKF_state = pd.DataFrame(
        {'time': data_kf['time_arr_filter'], r"J$_y$": data_kf['jy_lin'], r"J$_z$": data_kf['jz_lin'],
         'q': data_kf['q_lin'], 'p': data_kf['p_lin']})
    EKF_state = pd.DataFrame(
        {'time': data_kf['time_arr_filter'], r"J$_y$": data_kf['jy_ext'], r"J$_z$": data_kf['jz_ext'],
         'q': data_kf['q_ext'], 'p': data_kf['p_ext']})
    plot_state_simulation(simulation_state,
                          filename='plt_state_simulation_gp_%r_wp_%r.png' % (gp, w_p),
                          target_dir=target_dir)
    plot_state_LKF(EKF_state,
                   simulation_state,
                   filename='plt_state_LKF_gp_%r_wp_%r.png' % (gp, w_p),
                   target_dir=target_dir)
    plot_state_LKF_EKF(LKF_state,
                       EKF_state,
                       simulation_state,
                       filename='plt_state_LKF_EKF_gp_%r_wp_%r.png' % (gp, w_p),
                       target_dir=target_dir)

    # PLOT STATE VECTOR ESTIMATION ERR
    err_cov_LKF = pd.DataFrame(
        {'time':  data_kf['time_arr_filter'],
         r"$\Delta^2$J$_y$": data_kf['jy_err_lin_cov'],
         r"$\Delta^2$J$_z$": data_kf['jz_err_lin_cov'],
         r"$\Delta^2$q": data_kf['q_err_lin_cov'],
         r"$\Delta^2$p": data_kf['p_err_lin_cov']})
    err_LKF = pd.DataFrame(
        {'time':  data_kf['time_arr_filter'],
         r"$\Delta^2$J$_y/\Delta^2$J$_y^{ss}$": np.divide(data_kf['jy_err_lin'], data_kf['jy_steady_err']),
         r"$\Delta^2$J$_z/\Delta^2$J$_z^{ss}$": np.divide(data_kf['jz_err_lin'], data_kf['jz_steady_err']),
         r"$\Delta^2$q$/\Delta^2$q$^{ss}$": np.divide(data_kf['q_err_lin'], data_kf['q_steady_err']),
         r"$\Delta^2$p$/\Delta^2$p$^{ss}$": np.divide(data_kf['p_err_lin'], data_kf['p_steady_err'])})
    err_cov_EKF = pd.DataFrame(
        {'time': data_kf['time_arr_filter'],
         r"$\Delta^2$J$_y$": data_kf['jy_err_ext_cov'],
         r"$\Delta^2$J$_z$": data_kf['jz_err_ext_cov'],
         r"$\Delta^2$q": data_kf['q_err_ext_cov'],
         r"$\Delta^2$p": data_kf['p_err_ext_cov']})
    err_EKF = pd.DataFrame(
        {'time': data_kf['time_arr_filter'],
         r"$\Delta^2$J$_y/\Delta^2$J$_y^{ss}$": np.divide(data_kf['jy_err_ext'], data_kf['jy_steady_err']),
         r"$\Delta^2$J$_z/\Delta^2$J$_z^{ss}$": np.divide(data_kf['jz_err_ext'], data_kf['jz_steady_err']),
         r"$\Delta^2$q$/\Delta^2$q$^{ss}$": np.divide(data_kf['q_err_ext'], data_kf['q_steady_err']),
         r"$\Delta^2$p$/\Delta^2$p$^{ss}$": np.divide(data_kf['p_err_ext'], data_kf['p_steady_err'])})
    err_ss = pd.DataFrame(
        {'time':  data_kf['time_arr_filter'],
         r"$\Delta^2$J$_y$": data_kf['jy_steady_err'],
         r"$\Delta^2$J$_z$": data_kf['jz_steady_err'],
         r"$\Delta^2$q": data_kf['q_steady_err'],
         r"$\Delta^2$p": data_kf['p_steady_err']})
    err_ss_div = pd.DataFrame(
        {'time':  data_kf['time_arr_filter'],
         r"$\Delta^2$J$_y/\Delta^2$J$_y^{ss}$": np.divide(data_kf['jy_steady_err'], data_kf['jy_steady_err']),
         r"$\Delta^2$J$_z/\Delta^2$J$_z^{ss}$": np.divide(data_kf['jz_steady_err'], data_kf['jz_steady_err']),
         r"$\Delta^2$q$/\Delta^2$q$^{ss}$": np.divide(data_kf['q_steady_err'], data_kf['q_steady_err']),
         r"$\Delta^2$p$/\Delta^2$p$^{ss}$": np.divide(data_kf['p_steady_err'], data_kf['p_steady_err'])})
    plot_state_err_cov_LKF(err_cov_LKF,
                       err_ss,
                       filename='plt_state_err_cov_LKF_gp_%r_wp_%r.png' % (
                       gp, w_p),
                       target_dir=target_dir)
    plot_state_err_LKF(err_LKF,
                       err_ss_div,
                       filename='plt_state_err_LKF_gp_%r_wp_%r.png' % (
                       gp, w_p),
                       target_dir=target_dir)
    plot_state_err_LKF_EKF(err_LKF,
                           err_EKF,
                           err_ss_div,
                           filename='plt_state_err_LKF_EKF_gp_%r_wp_%r.png' % (
                               gp, w_p),
                           target_dir=target_dir
                           )

    # #PLOT REAL ESTIMATION ERR SQ(np.dot((x-x_est), (x-x_est).T))
    # err_LKF = pd.DataFrame(
    #     {'time':  data_kf['time_arr_filter'],
    #      r"$\Delta^2$J$_y$": error_jy_LKF,
    #      r"$\Delta^2$J$_z$": error_jz_LKF,
    #      r"$\Delta^2$q": error_q_LKF,
    #      r"$\Delta^2$p": error_p_LKF})
    # # err_EKF = pd.DataFrame(
    # #     {'time':  data_kf['time_arr_filter'],
    # #      r"$\Delta^2$J$_y$": error_jy_EKF,
    # #      r"$\Delta^2$J$_z$": error_jz_EKF,
    # #      r"$\Delta^2$q": error_q_EKF,
    # #      r"$\Delta^2$p": error_p_EKF})
    # plot_state_real_err_LKF(err_LKF,
    #                         err_cov,
    #                         filename='plt_state_real_err_gp_%r_wp_%r.png' % (gp, w_p))

    # PLOT zs/sigma
    # zs_real = pd.DataFrame(
    #     {
    #         'time':  data_kf['time_arr_filter'],
    #         "$z_k/\sigma_D$": zs_sigma
    #     }
    # )
    # zs_est_LKF = pd.DataFrame(
    #     {
    #         'time':  data_kf['time_arr_filter'],
    #         "$z_k/\sigma_D$": lkf_num_history_manager.zs_est
    #     }
    # )
    # plot_zs(zs_real, zs_est_LKF, 'plt_zs_gp_%r_wp_%r.png' % (gp, w_p),
    #         target_dir=target_dir)

    # PLOT WAVEFORM
    waveform_real = pd.DataFrame(
        {
            'time':  data_sim['time'],
            r"$\varepsilon$": data_sim['waveform']
        }
    )
    waveform_LKF = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            r"$\varepsilon$": data_kf['waveform_est_LKF']
        }
    )
    waveform_EKF = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            r"$\varepsilon$": data_kf['waveform_est_EKF']
        }
    )

    waveform_err_real_LKF = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            r"$\Delta^2 \varepsilon / \Delta^2 \varepsilon^{ss}$": np.divide(data_kf['waveform_real_LKF_err'], data_kf['waveform_ss_error'])
        }
    )
    waveform_err_cov_LKF = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            r"$\Delta^2 \varepsilon$": data_kf['waveform_est_LKF']
        }
    )
    waveform_err_cov_EKF = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            r"$\Delta^2 \varepsilon$": data_kf['waveform_est_EKF']
        }
    )
    waveform_err_real_EKF = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            r"$\Delta^2 \varepsilon / \Delta^2 \varepsilon^{ss}$": np.divide(data_kf['waveform_real_EKF_err'], data_kf['waveform_ss_error'])
        }
    )
    waveform_err_ss_div = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            r"$\Delta^2 \varepsilon / \Delta^2 \varepsilon^{ss}$": np.divide(data_kf['waveform_ss_error'], data_kf['waveform_ss_error'])
        }
    )
    plot_waveform_simulation(waveform_real,
                             'plt_waveform_simulation_gp_%r_wp_%r.png' % (gp, w_p),
                             target_dir=target_dir
                             )
    plot_waveform(waveform_real, waveform_LKF, waveform_err_real_LKF, waveform_err_ss_div,
                  'plt_waveform_gp_%r_wp_%r.png' % (gp, w_p),
                  target_dir=target_dir)
    plot_waveform_EKF_LKF(waveform_real,
                          waveform_LKF,
                          waveform_EKF,
                          waveform_err_real_LKF,
                          waveform_err_real_EKF,
                          waveform_err_ss_div,
                          'plt_waveform_gp_%r_wp_%r_LKF_EKF.png' % (gp, w_p),
                          target_dir=target_dir
                          )

    # PLOT DIFFERENCE LKF-EKF ESTIMATES
    diff_LKF_EKF = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            'diff': np.abs(data_kf['q_lin'] -data_kf['q_ext'])
        })

    diff_LKF_EKF_err = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            'diff': np.abs(
                np.sqrt(data_kf['q_err_lin_cov']) - np.sqrt(data_kf['q_err_ext_cov']))
        })
    plot_q_est_difference_LKF_EKF(diff_LKF_EKF, diff_LKF_EKF_err, 'plt_q_LKF_EKF_difference_gp_%r_wp_%r.png' % (
        gp, w_p), target_dir=target_dir)

    # # PLOT DIFFERENCE LKF-num_exp ESTIMATES
    # diff_LKF_num_exp = pd.DataFrame(
    #     {
    #         'time':  data_kf['time_arr_filter'],
    #         'diff': np.abs(data_kf['q_lin'] - data_kf['q_lin_approx'])
    #     })
    # diff_LKF_EKF_num_exp_err = pd.DataFrame(
    #     {
    #         'time':  data_kf['time_arr_filter'],
    #         'diff': np.abs(
    #             np.sqrt(data_kf['q_err_lin_cov']) - np.sqrt(data_kf['q_err_lin_approx_cov']))
    #     })
    # plot_q_est_difference_num_exp(diff_LKF_num_exp, diff_LKF_EKF_num_exp_err,
    #                               'plt_LKF_num_exp_difference_gp_%r_wp_%r.png' % (
    #                                   gp, w_p), target_dir=target_dir)

    # PLOT EKF LIN AND DISC
    ekf_lin = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            'q':data_kf['q_ext_lin'],
            r"$\Delta^2$q": data_kf['q_err_ext_lin_cov']
        })
    ekf_disc = pd.DataFrame(
        {
            'time':  data_kf['time_arr_filter'],
            'q':data_kf['q_ext'],
            r"$\Delta^2$q": data_kf['q_err_ext_cov']
        })
    err_ss_q = pd.DataFrame(
        {'time':  data_kf['time_arr_filter'],
         r"$\Delta^2$q": data_kf['q_steady_err']})
    sim_q = pd.DataFrame({'time':  data_sim['time'], 'q': data_sim['q']})

    plot_ekf_lin_disc(ekf_lin, ekf_disc, err_ss_q, sim_q,
                      'plt_EKF_lin_disc_gp_%r_wp_%r.png' % (
                          gp, w_p), target_dir=target_dir)

gps = np.arange(5, 170, 20).tolist()
for gp in gps:
    wp = 6.
    target_dir = "./Simulation_plots/gp_%r_wp_%r" % (int(gp), int(wp))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    plot__all_atomic_sensor_from_files(gp, wp, target_dir=target_dir)

wps = np.arange(0., 10., 1.).tolist()
for wp in wps:
    gp = 145
    target_dir = "./Simulation_plots/gp_%r_wp_%r" % (int(gp), int(wp))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    plot__all_atomic_sensor_from_files(gp, wp, target_dir=target_dir)

