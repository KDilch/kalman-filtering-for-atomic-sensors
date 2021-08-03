from atomic_sensor_simulation.history_manager.history_manager import HistoryManager


class KalmanFilterHistoryManager(HistoryManager):

    def __init__(self):
        HistoryManager.__init__(self)
        self.__state_estimates_data = []
        self.__covariance_priors_data = []
        self.__covariance_posts_data = []
        self.__time_arr = []

    def add_history_point(self, history_point):
        """
        :param history_point: should be a key value pair (time: value of the measurement)
        :return:
        """
        self.__time_arr.append(history_point[0])
        self.__state_estimates_data.append(history_point[1])
        self.__covariance_priors_data.append(history_point[2])
        self.__covariance_posts_data.append(history_point[3])

    @property
    def full_history(self):
        return self.__state_estimates_data

    @property
    def full_history_cov_posts(self):
        return self.__covariance_posts_data

    @property
    def full_history_cov_priors(self):
        return self.__covariance_priors_data

    @property
    def time_arr(self):
        return self.__time_arr
