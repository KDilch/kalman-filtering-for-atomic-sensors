from atomic_sensor_simulation.history_manager.history_manager import HistoryManager

import matplotlib.pyplot as plt
import numpy as np
import os


class AtomicSensorSimulationHistoryManager(HistoryManager):
    def __init__(self):
        HistoryManager.__init__(self)
        self.__simulation_data = []
        self.__simulation_data_measurement_frequency = []
        self.__time_arr = []
        self.__time_arr_measurement_frequency = []

    def add_history_point(self, history_point, is_measurement_performed=False):
        """
        :param is_measurement_performed:
        :param history_point: should be a key value pair
        :return:
        """
        self.__time_arr.append(history_point[0])
        self.__simulation_data.append(history_point[1])
        if is_measurement_performed:
            self.__time_arr_measurement_frequency.append(history_point[0])
            self.__simulation_data_measurement_frequency.append(history_point[1])
        return

    @property
    def full_simulation_history(self):
        return self.__simulation_data

    @property
    def time_arr(self):
        return self.__time_arr
