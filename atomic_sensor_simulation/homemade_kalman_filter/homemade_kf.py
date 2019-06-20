#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import solve_discrete_are


class HomeMadeKalmanFilter(object):

    def __init__(self, x0, P0, Phi_delta, Q_delta, H, R_delta, G=None):
        self.x = x0
        self.P = P0
        self.Phi_delta = Phi_delta
        self.dim_x = len(x0)
        self.dim_z = None #needed to perform checks
        self.Q_delta = Q_delta
        self.H = H
        self.R_delta = R_delta
        if G:
            self.G = G
        else:
            self.G = np.identity(self.dim_x)

    def predict(self):
        x = np.dot(self.Phi_delta, self.x)
        P = np.dot(np.dot(self.Phi_delta, self.P), np.transpose(self.Phi_delta)) + self.Q_delta
        self.x = x
        self.P = P

    def update(self, z):
        z_est = np.dot(self.H, self.x)
        y = z - z_est
        S = self.R_delta + np.dot(np.dot(self.H, self.P), np.transpose(self.H))
        S_inverse = np.linalg.inv(S)
        K = np.dot(np.dot(self.P, np.transpose(self.H)), S_inverse)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.identity(2) - np.dot(K, self.H)), self.P)

    def steady_state(self):
        steady_cov_predict = solve_discrete_are(a=np.transpose(self.Phi_delta),
                                                b=np.transpose(self.H),
                                                r=self.R_delta,
                                                q=self.Q_delta)
        S_steady = self.R_delta + np.dot(np.dot(self.H, steady_cov_predict), np.transpose(self.H))
        K_steady = np.dot(np.dot(steady_cov_predict, np.transpose(self.H)), np.linalg.inv(S_steady))
        steady_cov_update = np.dot(np.identity(self.dim_x)-np.dot(K_steady, self.H), steady_cov_predict)
        return steady_cov_predict, steady_cov_update
