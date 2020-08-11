# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from atomic_sensor_simulation.utilities import generate_data_arr_for_saving

plt.style.use('seaborn-darkgrid')

def plot_w_p(gp, filename, target_dir='./'):
    palette = plt.get_cmap('Set1')

    wps = np.arange(1., 10., 1.).tolist()
    # wps = [1., 3., 6., 9.]
    wp = wps[0]

    plt.figure(figsize=(14, 9))

    data_sim = pd.read_csv("./Simulation_data/data_sim_gp_%r_wp_%r.csv" % (gp, wp), sep='\t', usecols=['time', 'q'])

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(data_sim['time'][::30], data_sim['q'][::30], linestyle='none', marker='.',
             color='grey', label='Simulation')
    ax2 = plt.subplot(1, 2, 2)

    if filename is None:
        filename = 'plt_omega_gp%r' % gp

    num = 0
    omega_ratios = []
    maximums = []
    from scipy.optimize import curve_fit
    def func(x, a, c, d):
        return a * np.exp(- c * x) + d

    for wp in wps:
        data_kf = pd.read_csv("./Simulation_data/data_kf_gp_%r_wp_%r.csv" % (gp, wp), sep='\t',
                              usecols=['time_arr_filter', 'q_lin', 'q_err_lin_cov', 'q_steady_err'])
        omega_ratios.append(wp/6.)
        maximums.append(max(data_kf['q_err_lin_cov'][100:]))
    popt, pcov = curve_fit(func, omega_ratios, maximums)
    for wp in wps:
        omega_ratio = wp/6.  # Larmour freq is 6.
        num += 1
        data_kf = pd.read_csv("./Simulation_data/data_kf_gp_%r_wp_%r.csv" % (gp, wp), sep='\t',
                              usecols=['time_arr_filter', 'q_lin', 'q_err_lin_cov', 'q_steady_err'])
        ax1.plot(data_kf['time_arr_filter'], data_kf['q_lin'], marker='', linewidth=1.9, alpha=0.9,
                 label="$\omega_p$=%r_a" % wp, color=palette(num))
        ax2.plot([omega_ratio], max(data_kf['q_err_lin_cov'][100:]), marker='o', linestyle='none', alpha=0.9,
                 label="$\omega_p/\omega_L=%.1f$" % omega_ratio, color=palette(num))
        # ax2.plot(data_kf['time_arr_filter'], data_kf['q_steady_err'], marker='', color=palette(num), linewidth=1.9,
        #          alpha=0.9, linestyle='--')

            # plt.xlabel('time [a.u.]', fontsize=18)
            # plt.ylabel(column + ' [a.u.]', fontsize=18)

        # ax2.legend(fontsize=18, fancybox=True, framealpha=0.5)
    ax2.plot(np.arange(0, 2, 0.01), func(np.arange(0, 2, 0.01), *popt))
    # ax2.set_yscale('log', basey=10)
    ax1.set_xlabel('time [a.u.]', fontsize=18)
    ax1.set_ylabel('q [a.u.]', fontsize=18)
    ax2.set_xlabel("$\omega_p/\omega_L$", fontsize=18)
    ax2.set_ylabel("max $\Delta^2 q$ [a.u.]", fontsize=18)
    # ax2.set_ylim([None, 0.0006])

    plt.savefig(os.path.join(target_dir, filename))
    print(*popt, *pcov)
    fitting_errs = np.sqrt(np.abs(np.diag(pcov)))
    data = generate_data_arr_for_saving([popt, fitting_errs], ["param", "err"], [True, True])
    filename = filename + "_fit_params"
    data.to_csv(os.path.join(target_dir, filename))
    plt.show()
    # plt.legend()
    plt.close()

gp = 145
plot_w_p(gp=gp, filename=None)