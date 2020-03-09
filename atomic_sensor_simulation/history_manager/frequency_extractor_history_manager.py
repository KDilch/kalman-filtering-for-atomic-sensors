import numpy as np

from utilities import compute_squred_error_from_covariance

class Filter_History_Manager(object):
    def __init__(self, filter_obj, num_iter_filter):
        self.__filter_obj = filter_obj
        self.__x1s = np.zeros(num_iter_filter)
        self.__x1s_err_prior = np.zeros(num_iter_filter)
        self.__x1s_err_post = np.zeros(num_iter_filter)
        self.__x2s = np.zeros(num_iter_filter)
        self.__x2s_err_prior = np.zeros(num_iter_filter)
        self.__x2s_err_post = np.zeros(num_iter_filter)
        self.__x3s = np.zeros(num_iter_filter)
        self.__x3s_err_prior = np.zeros(num_iter_filter)
        self.__x3s_err_post = np.zeros(num_iter_filter)

    @property
    def x1s(self):
        return self.__x1s

    @property
    def x1s_err_prior(self):
        return self.__x1s_err_prior

    @property
    def x1s_err_post(self):
        return self.__x1s_err_post

    @property
    def x2s(self):
        return self.__x2s

    @property
    def x2s_err_prior(self):
        return self.__x2s_err_prior

    @property
    def x2s_err_post(self):
        return self.__x1s_err_post

    @property
    def x3s(self):
        return self.__x3s

    @property
    def x3s_err_prior(self):
        return self.__x3s_err_prior

    @property
    def x3s_err_post(self):
        return self.__x3s_err_post

    def add_entry(self, index):
        self.__x1s[index] = self.__filter_obj.x[0]
        self.__x2s[index] = self.__filter_obj.x[1]
        self.__x3s[index] = self.__filter_obj.x[2]
        self.__x1s_err_prior[index] = compute_squred_error_from_covariance(self.__filter_obj.P_prior, index=0)
        self.__x2s_err_prior[index] = compute_squred_error_from_covariance(self.__filter_obj.P_prior, index=1)
        self.__x3s_err_prior[index] = compute_squred_error_from_covariance(self.__filter_obj.P_prior, index=2)
        self.__x1s_err_post[index] = compute_squred_error_from_covariance(self.__filter_obj.P_post, index=0)
        self.__x2s_err_post[index] = compute_squred_error_from_covariance(self.__filter_obj.P_post, index=1)
        self.__x3s_err_post[index] = compute_squred_error_from_covariance(self.__filter_obj.P_post, index=2)
