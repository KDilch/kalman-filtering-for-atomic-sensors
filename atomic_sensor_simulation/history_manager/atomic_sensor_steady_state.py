from atomic_sensor_simulation.history_manager.history_manager import HistoryManager


class AtomicSensorSteadyStateHistoryManager(HistoryManager):
    def __init__(self):
        HistoryManager.__init__(self)
        self.__steady_state_prior_data = []
        self.__steady_state_post_data = []
        self.__time_arr = []

    def add_history_point(self, history_point):
        """
        :param history_point: should be a key value pair
        :return:
        """
        self.__time_arr.append(history_point[0])
        self.__steady_state_prior_data.append(history_point[1][0])
        self.__steady_state_post_data.append(history_point[1][1])
        return

    @property
    def full_history(self):
        return self.__steady_state_post_data

    @property
    def time_arr(self):
        return self.__time_arr
