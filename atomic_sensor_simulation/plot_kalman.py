import matplotlib.pyplot as plt
import numpy as np
import os


def plot_simulation_and_kalman(simulation_data, kalman_data, jy=True, jz=True, p=True, q=True, show=False, output_file=None):
    if (not show) & (not output_file):
        raise UserWarning('Plotting not performed as show parameter is set to False and the output file is None.')
    temp = np.array(simulation_data.full_simulation_history)
    temp_kalman = np.array(kalman_data.full_history)
    if jy:
        plt.plot(simulation_data.time_arr, temp[:, 0], label='simulation')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 0], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()

    if jz:
        plt.plot(simulation_data.time_arr, temp[:, 1], label='simulation')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 1], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()

    if p:
        plt.plot(simulation_data.time_arr, temp[:, 2], label='simulation')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 2], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()

    if q:
        plt.plot(simulation_data.time_arr, temp[:, 3], label='simulation')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 3], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()


def plot_kalman_and_steady_state(kalman_data, steady_state_data, jy=True, jz=True, p=True, q=True, show=False, output_file=None):
    if (not show) & (not output_file):
        raise UserWarning('Plotting not performed as show parameter is set to False and the output file is None.')
    temp_steady_state = np.array(steady_state_data.full_history)
    temp_kalman = np.array(kalman_data.full_history_cov_posts)
    if jy:
        plt.plot(steady_state_data.time_arr, temp_steady_state[:, 0, 0], label='steady state')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 0, 0], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()

    if jz:
        plt.plot(steady_state_data.time_arr, temp_steady_state[:, 1, 1], label='steady state')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 1, 1], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()

    if p:
        plt.plot(steady_state_data.time_arr, temp_steady_state[:, 2, 2], label='steady state')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 2, 2], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()

    if q:
        plt.plot(steady_state_data.time_arr, temp_steady_state[:, 3, 3], label='steady_state')
        plt.plot(kalman_data.time_arr, temp_kalman[:, 3, 3], label='kf')
        plt.legend()
        if show:
            plt.show()
        if output_file:
            plt.savefig(os.path.join(output_file))
        plt.close()
