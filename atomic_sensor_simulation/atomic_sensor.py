# -*- coding: utf-8 -*-
import numpy as np
from noise import GaussianWhiteNoise


class Spin(object):

    def __init__(self, larmor_freq, t2_param, scalar_strength):
        self.__precession = Precession(larmor_freq, t2_param)
        self.__noise = GaussianWhiteNoise(scalar_strength)
        self.__signal = None

        self.__vec_previous = None
        self.__vec_current = None

    @property
    def spin_vec(self):
        return self.__vec_current

    def update(self, time_step):
        self.__precession.step(self.__vec_current, time_step)
        self.__signal.step()
        self.__vec_previous = self.__vec_current
        self.__vec_current = self.__precession.val


class Precession(object):

    def __init__(self, larmor_freq, t2_param):
        self.__larmor_freq = larmor_freq
        self.__t2_param = t2_param
        self.__arr = np.array([-1/self.__t2_param, self.__larmor_freq], [-self.__larmor_freq, -1/self.__t2_param])
        self.__val = None

    @property
    def t2_param(self):
        return self.__t2_param

    @property
    def larmor_freq(self):
        return self.__larmor_freq

    @property
    def arr(self):
        return self.__arr

    @property
    def val(self):
        if self.__val is not None:
            return self.__val
        else:
            Warning('Not current value defined for precession. You need to step first.')

    def step(self, spin_vec, time_step):
        assert type(spin_vec) == type(np.array([]))
        self.__val = self.__arr.dot(spin_vec) * time_step
        return

class Signal(object):

    def __init__(self, larmor_freq, quadrature, coupling_const, correlation_time, scalar_strength):
        self.__larmor_freq = larmor_freq
        self.__quadrature = OrnsteinUhlenbeckProcess(quadrature, correlation_time, scalar_strength)
        self.__coupling_const = coupling_const
        self.__val = None
        self.__noise = None

    @property
    def val(self):
        if self.__val is not None:
            return self.__val
        else:
            Warning('Not current value defined for signal. You need to step first.')

    def step(self, time, time_step):
        self.__quadrature.step(time_step)
        self.__val = np.array([0, self.__func(time+time_step).dot(self.__quadrature.val)])
        return

    def __func(self, time):
        return np.array([self.__coupling_const*np.cos(self.__larmor_freq*time), self.__coupling_const*np.sin(self.__larmor_freq*time)])


class OrnsteinUhlenbeckProcess(object):

    def __init__(self, quadrature, correlation_time, scalar_strength):
        self.__correlation_time = correlation_time
        self.__quadrature = quadrature
        self.__noise = GaussianWhiteNoise(scalar_strength)
        self.__val = None

    @property
    def val(self):
        if self.__val is not None:
            return self.__val
        else:
            Warning('Not current value defined for quadrature. You need to step first.')

    def step(self, time_step):
        self.__noise.step(time_step)
        self.__val = -self.__correlation_time*self.__quadrature * time_step + self.__noise.val
        return
