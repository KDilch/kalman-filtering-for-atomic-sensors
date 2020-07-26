# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-darkgrid')

def plot_g_p(w_p, filename, target_dir='./'):
    palette = plt.get_cmap('Set1')

    gps = np.arange(5, 150, 20).tolist()
    gp = gps[0]
    plt.figure(figsize=(14, 9))

    data_sim = pd.read_csv("./Simulation_data/data_sim_gp_%r_wp_%r.csv" % (gp, w_p), sep='\t', usecols=['time', 'q'])

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(data_sim['time'][::30], data_sim['q'][::30], linestyle='none', marker='.',
             color='grey', label='Simulation')
    ax2 = plt.subplot(1, 2, 2)

    num = 0
    for gp in gps:
        num += 1
        data_kf = pd.read_csv("./Simulation_data/data_kf_gp_%r_wp_%r.csv" % (gp, w_p), sep='\t',
                              usecols=['time_arr_filter', 'q_lin', 'q_err_lin_cov', 'q_steady_err'])

        ax1.plot(data_kf['time_arr_filter'], data_kf['q_lin'], marker='', linewidth=1.9, alpha=0.9,
                 label="$g_p$=%r_a" % gp, color=palette(num))
        ax2.plot(data_kf['time_arr_filter'], data_kf['q_err_lin_cov'], marker='', linewidth=1.9, alpha=0.9,
                 label="$g_p$=%r" % gp, color=palette(num))
        ax2.plot(data_kf['time_arr_filter'], data_kf['q_steady_err'], marker='', color=palette(num), linewidth=1.9,
                 alpha=0.9, linestyle='--')
            #
            # plt.xlabel('time [a.u.]', fontsize=18)
            # plt.ylabel(column + ' [a.u.]', fontsize=18)
            #
        ax2.legend(fontsize=18)

    # ax2.set_yscale('log', basey=10)
    ax1.set_xlabel('time [a.u.]', fontsize=18)
    ax1.set_ylabel('q [a.u.]', fontsize=18)
    ax2.set_xlabel('time [a.u.]', fontsize=18)
    ax2.set_ylabel("$\Delta^2 q$ [a.u.]", fontsize=18)
    plt.savefig(os.path.join(target_dir, filename))
    plt.show()
    # plt.legend()
    plt.close()

plot_g_p(w_p=0.,filename='gp_wp_0.png')