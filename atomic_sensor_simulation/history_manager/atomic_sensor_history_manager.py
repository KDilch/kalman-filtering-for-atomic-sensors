import numpy as np

from utilities import compute_squred_error_from_covariance

class Filter_History_Manager(object):
    def __init__(self, filter_obj, num_iter_filter, config, time):
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
        self.__zs_est = np.zeros(num_iter_filter)
        self.__waveform_est = np.zeros(num_iter_filter)
        self.__waveform_est_err = np.zeros(num_iter_filter)
        self.__time = time
        self.config = config
        self.sigma_D = np.sqrt(self.config.noise_and_measurement['QD']/self.config.filter['dt_filter'])

    @property
    def time(self):
        return self.__time

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

    @property
    def zs_est(self):
        return self.__zs_est

    @property
    def waveform_est(self):
        return self.__waveform_est

    @property
    def waveform_est_err(self):
        return self.__waveform_est_err

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
        self.__zs_est[index] = self.config.noise_and_measurement['gD']*self.__jzs[index]/self.sigma_D
        self.__waveform_est[index] = self.config.coupling['g_p']*(np.cos(self.config.coupling['omega_p']*self.__time[index])*self.__qs[index]+np.sin(self.config.coupling['omega_p']*self.__time[index])*self.__ps[index])
        self.__waveform_est_err[index] = (self.config.coupling['g_p'])**2*((((np.cos(self.config.coupling['omega_p']*self.__time[index]))**2)*self.__qs_err_post[index])+(((np.sin(self.config.coupling['omega_p']*self.__time[index])**2)*(self.__ps_err_post[index]))))

class SteadyStateHistoryManager(object):
    def __del__(self):
        print("Deleting SteadyStateHistoryManager")

    def __init__(self, num_iter_filter, config, time):
        print("Creating SteadyStateHistoryManager")
        self.__steady_priors_jy = np.zeros(num_iter_filter)
        self.__steady_posts_jy = np.zeros(num_iter_filter)
        self.__steady_priors_jz = np.zeros(num_iter_filter)
        self.__steady_posts_jz = np.zeros(num_iter_filter)
        self.__steady_priors_q = np.zeros(num_iter_filter)
        self.__steady_posts_q = np.zeros(num_iter_filter)
        self.__steady_priors_p = np.zeros(num_iter_filter)
        self.__steady_posts_p = np.zeros(num_iter_filter)
        self.__steady_waveform_err = np.zeros(num_iter_filter)
        self.config = config
        self.__time = time

    @property
    def steady_priors_jy(self):
        return self.__steady_priors_jy

    @property
    def steady_priors_jz(self):
        return self.__steady_priors_jz

    @property
    def steady_priors_q(self):
        return self.__steady_priors_q

    @property
    def steady_priors_p(self):
        return self.__steady_priors_p

    @property
    def steady_posts_jy(self):
        return self.__steady_posts_jy

    @property
    def steady_posts_jz(self):
        return self.__steady_posts_jz

    @property
    def steady_posts_q(self):
        return self.__steady_posts_q

    @property
    def steady_posts_p(self):
        return self.__steady_posts_p

    @property
    def steady_waveform_err(self):
        return self.__steady_waveform_err

    def add_entry(self, steady_prior, steady_post, index):
        self.__steady_priors_jy[index] = compute_squred_error_from_covariance(steady_prior, 0)
        self.__steady_priors_jz[index] = compute_squred_error_from_covariance(steady_prior, 1)
        self.__steady_priors_q[index] = compute_squred_error_from_covariance(steady_prior, 2)
        self.__steady_priors_p[index] = compute_squred_error_from_covariance(steady_prior, 3)
        self.__steady_posts_jy[index] = compute_squred_error_from_covariance(steady_post, 0)
        self.__steady_posts_jz[index] = compute_squred_error_from_covariance(steady_post, 1)
        self.__steady_posts_q[index] = compute_squred_error_from_covariance(steady_post, 2)
        self.__steady_posts_p[index] = compute_squred_error_from_covariance(steady_post, 3)
        self.__steady_waveform_err[index] = self.config.coupling['g_p']**2*((((np.cos(self.config.coupling['omega_p']*self.__time[index]))**2)*self.__steady_posts_q[index])+(((np.sin(self.config.coupling['omega_p']*self.__time[index])**2)*(self.__steady_posts_p[index]))))
