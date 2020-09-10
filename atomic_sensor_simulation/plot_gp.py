# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.optimize import curve_fit

from utilities import generate_data_arr_for_saving

plt.style.use('seaborn-darkgrid')

def plot_g_p(w_p, filename, target_dir='./'):
    palette = plt.get_cmap('Set1')

    gps = np.arange(5, 170, 20)
    gp = gps[0]
    plt.figure(figsize=(12, 6))

    data_sim = pd.read_csv("./Simulation_data/data_sim_gp_%r_wp_%r.csv" % (gp, w_p), sep='\t', usecols=['time', 'q'])

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(data_sim['time'][::30], data_sim['q'][::30], linestyle='none', marker='.',
             color='grey', label='Simulation')
    ax2 = plt.subplot(1, 2, 2)
    maximums = []
    maximums_ext = []
    def func(x, a, b, c):
        return a * np.exp(b*x) + c

    def lin_func(x, a, b):
        return a*x + b

    for gp in gps:
        data_kf = pd.read_csv("./Simulation_data/data_kf_gp_%r_wp_%r.csv" % (gp, w_p), sep='\t',
                              usecols=['time_arr_filter', 'q_lin', 'q_err_lin_cov', 'q_ext', 'q_err_ext_cov', 'q_steady_err'])
        maximums.append(max(data_kf['q_err_lin_cov'][100:]))
        maximums_ext.append(max(data_kf['q_err_ext_cov'][100:]))

    # a = np.polyfit(gps, maximums, 4, cov=True)
    # polynomial = np.poly1d(a[0])
    # print('fit params', a[0])
    # print('cov', a[1])
    # model = np.poly1d(polynomial)

    import statsmodels.formula.api as smf

    df = pd.DataFrame(columns=['y', 'x'])
    df['x'] = gps
    df['y'] = maximums

    popt, pcov = curve_fit(func, gps, maximums, [0.00244301, -0.0553479, 0.00139758])
    popt_lin, pcov_lin = curve_fit(lin_func, gps, maximums_ext)
    print(popt_lin, pcov_lin)
    model = func(gps, *popt)
    model_lin = lin_func(gps, *popt_lin)
    results = smf.ols(formula='maximums_ext ~ model_lin', data=df).fit()
    print(results.summary())
    num = 0
    for index, gp in enumerate(gps):
        num += 1
        data_kf = pd.read_csv("./Simulation_data/data_kf_gp_%r_wp_%r.csv" % (gp, w_p), sep='\t',
                              usecols=['time_arr_filter', 'q_lin', 'q_err_ext_cov', 'q_ext', 'q_err_ext_cov', 'q_steady_err'])
        if num % 2:
            ax1.plot(data_kf['time_arr_filter'], data_kf['q_ext'], marker='', linewidth=1.9, alpha=0.9,
                     label="$g_p$=%r" % gp, color=palette(num))
        ax2.plot(gp, maximums_ext[index], marker='o', linestyle='none', alpha=0.9,
                 label="$g_p$=%r_a" % gp, color=palette(num))
        # ax2.set_yscale('log')
    ax2.plot(np.arange(0, 170, 1.), func(np.arange(0, 170, 1.), *popt), linestyle='dashdot')
    # ax2.plot(np.arange(0, 170, 1.), lin_func(np.arange(0, 170, 1.), *popt_lin), linestyle='dashdot')

    # ax2.plot(np.arange(1, 170, 1.), polynomial(np.arange(1, 170, 1.)), linestyle='dashdot')

        # ax2.plot(data_kf['time_arr_filter'], data_kf['q_steady_err'], marker='', color=palette(num), linewidth=1.9,
        #         #          alpha=0.9, linestyle='--')
            #
            # plt.xlabel('time [a.u.]', fontsize=18)
            # plt.ylabel(column + ' [a.u.]', fontsize=18)
            #
    ax1.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)

    # ax2.set_yscale('log', basey=10)
    ax1.set_xlabel('time [a.u.]', fontsize=18)
    ax1.set_ylabel('q [a.u.]', fontsize=18)
    ax2.set_xlabel("$g_p$ [a.u.]", fontsize=18)
    ax2.set_ylabel("max $\Delta^2 q$ [a.u.]", fontsize=18)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.9, hspace=None)
    plt.savefig(os.path.join(target_dir, filename))
    fitting_errs = np.sqrt(np.abs(np.diag(pcov)))
    fitting_errs_lin = np.sqrt(np.abs(np.diag(pcov_lin)))
    print(fitting_errs_lin)
    data = generate_data_arr_for_saving([popt, fitting_errs], ["param", "err"], [True, True])
    filename = filename + "_fit_params"
    data.to_csv(os.path.join(target_dir, filename))
    plt.show()
    # plt.legend()
    plt.close()

plot_g_p(w_p=6.,filename='gp_wp_6.png')