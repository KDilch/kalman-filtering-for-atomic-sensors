from atomic_sensor_simulation.history_manager.history_manager import HistoryManager

import matplotlib.pyplot as plt
import numpy as np
import os


class AtomicSensorSimulationHistoryManager(HistoryManager):
    def __init__(self):
        HistoryManager.__init__(self)
        self.__simulation_data = []
        self.__time_arr = []

    def add_history_point(self, history_point):
        """
        :param history_point: should be a key value pair
        :return:
        """
        self.__time_arr.append(history_point[0])
        self.__simulation_data.append(history_point[1])

    def plot(self, jy=True, jz=True, p=True, q=True, show=False, output_file=None):
        if (not show) & (not output_file):
            raise UserWarning('Plotting not performed as show parameter is set to False and the output file is None.')
        temp = np.array(self.__simulation_data)
        if jy:
            plt.plot(self.__time_arr, temp[:, 0])
            if show:
                plt.show()
            if output_file:
                plt.savefig(os.path.join(output_file))
            plt.close()

        if jz:
            plt.plot(self.__time_arr, temp[:, 1])
            if show:
                plt.show()
            if output_file:
                plt.savefig(os.path.join(output_file))
            plt.close()

        if p:
            plt.plot(self.__time_arr, temp[:, 2])
            if show:
                plt.show()
            if output_file:
                plt.savefig(os.path.join(output_file))
            plt.close()

        if q:
            plt.plot(self.__time_arr, temp[:, 3])
            if show:
                plt.show()
            if output_file:
                plt.savefig(os.path.join(output_file))
            plt.close()
