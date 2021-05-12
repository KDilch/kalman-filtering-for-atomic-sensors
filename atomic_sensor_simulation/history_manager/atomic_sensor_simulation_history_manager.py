from atomic_sensor_simulation.history_manager.history_manager import HistoryManager

import matplotlib.pyplot as plt
import numpy as np


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

    def plot(self):
        temp = np.array(self.__simulation_data)
        plt.plot(self.__time_arr, temp[:, 3])
        plt.show()