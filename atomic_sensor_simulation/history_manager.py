import numpy as np

from utilities import compute_squred_error_from_covariance

class Filter_History_Manager(object):
    def __init__(self, filter_obj, num_iter_filter):
        self.__filter_obj = filter_obj
        self.__jys = np.zeros(num_iter_filter)
        self.__jys_err_prior = np.zeros(num_iter_filter)
        self.__jys_err_post = np.zeros(num_iter_filter)
        self.__jzs = np.zeros(num_iter_filter)
        self.__jzs_err_prior = np.zeros(num_iter_filter)
        self.__jzs_err_post = np.zeros(num_iter_filter)
        self.__qs = np.zeros(num_iter_filter)
        self.__qs_err_prior = np.zeros(num_iter_filter)
        self.__qs_err_post = np.zeros(num_iter_filter)
        self.__ps = np.zeros(num_iter_filter)
        self.__ps_err_prior = np.zeros(num_iter_filter)
        self.__ps_err_post = np.zeros(num_iter_filter)

    @property
    def jys(self):
        return self.__jys

    @property
    def jys_err_prior(self):
        return self.__jys_err_prior

    @property
    def jys_err_post(self):
        return self.__jys_err_post

    @property
    def jzs(self):
        return self.__jzs

    @property
    def jzs_err_prior(self):
        return self.__jzs_err_prior

    @property
    def jzs_err_post(self):
        return self.__jzs_err_post

    @property
    def qs(self):
        return self.__qs

    @property
    def qs_err_prior(self):
        return self.__qs_err_prior

    @property
    def qs_err_post(self):
        return self.__qs_err_post
    @property
    def ps(self):
        return self.__ps

    @property
    def ps_err_prior(self):
        return self.__ps_err_prior

    @property
    def ps_err_post(self):
        return self.__ps_err_post

    def add_entry(self, index):
        self.__jys[index] = self.__filter_obj.x[0]
        self.__jzs[index] = self.__filter_obj.x[1]
        self.__qs[index] = self.__filter_obj.x[2]
        self.__ps[index] = self.__filter_obj.x[3]
        self.__jys_err_prior[index] = compute_squred_error_from_covariance(self.__filter_obj.P_prior, index=0)
        self.__jzs_err_prior[index] = compute_squred_error_from_covariance(self.__filter_obj.P_prior, index=1)
        self.__qs_err_prior[index] = compute_squred_error_from_covariance(self.__filter_obj.P_prior, index=2)
        self.__ps_err_prior[index] = compute_squred_error_from_covariance(self.__filter_obj.P_prior, index=3)
        self.__jys_err_post[index] = compute_squred_error_from_covariance(self.__filter_obj.P_post, index=0)
        self.__jzs_err_post[index] = compute_squred_error_from_covariance(self.__filter_obj.P_post, index=1)
        self.__qs_err_post[index] = compute_squred_error_from_covariance(self.__filter_obj.P_post, index=2)
        self.__ps_err_post[index] = compute_squred_error_from_covariance(self.__filter_obj.P_post, index=3)

class StateHistoryManager(object):
    def __init__(self):
        def __init__(self, filter_obj, num_iter_filter):
            self.__filter_obj = filter_obj
            self.__jys = np.zeros(num_iter_filter)
            self.__jys_err = np.zeros(num_iter_filter)

    @property
    def jys(self):
        return self.__jys
