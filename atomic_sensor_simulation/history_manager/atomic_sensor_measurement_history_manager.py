import copy
from warnings import warn

from atomic_sensor_simulation.history_manager.history_manager import HistoryManager


class AtomicSensorMeasurementHistoryManager(HistoryManager):

    def __init__(self, is_store_all=False):
        HistoryManager.__init__(self)
        self.__measurement_data = []
        self.__time_arr = []
        self.__is_store_all = is_store_all
        self.__previous_point = None
        self.__previous_time = None
        self.__current_point = None
        self.__current_time = None

    def add_history_point(self, history_point):
        """
        :param history_point: should be a key value pair (time: value of the measurement)
        :return:
        """
        if self.__is_store_all:
            self.__time_arr.append(history_point[0])
            self.__measurement_data.append(history_point[1])
        self.__add_current_history_point(history_point)

    def is_the_first_measurement(self):
        if len(self.__measurement_data) == 1:
            return True
        else:
            return False

    def __add_current_history_point(self, current_history_point):
        self.__previous_time = copy.deepcopy(self.__current_time)
        self.__previous_point = copy.deepcopy(self.__current_point)
        self.__current_time = current_history_point[0]
        self.__current_point = current_history_point[1]
        return

    @property
    def current_history_point_value(self):
        return self.__current_point

    @property
    def previous_history_point_value(self):
        return self.__previous_point

    @property
    def current_time(self):
        return self.__current_time

    @property
    def previous_time(self):
        return self.__previous_time

    @property
    def full_history(self):
        if self.__measurement_data:
            return self.__measurement_data
        else:
            return warn('No history provided for this measurement scheme.', UserWarning)


class KalmanFilterEstimatesHistoryManager(AtomicSensorMeasurementHistoryManager):
    pass
