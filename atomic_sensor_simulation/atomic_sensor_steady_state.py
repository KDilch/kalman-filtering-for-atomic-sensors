#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import solve_discrete_are, expm


def compute_steady_state_solution_for_atomic_sensor(coupling_freq, coupling_phase_shift, t, F, model):

    F_RF = change_reference_frame_rotating(F, coupling_freq, coupling_phase_shift, t)
    Phi_delta_RF = expm(F_RF * model.dt)
    Q_delta = np.dot(np.dot(Phi_delta_RF, model.Q), Phi_delta_RF.transpose()) * model.dt
    steady_cov_predict_RF = solve_discrete_are(a=np.transpose(Phi_delta_RF),
                                            b=np.transpose(model.H),
                                            r=model.R_delta,
                                            q=Q_delta)
    S_steady = model.R_delta + np.dot(np.dot(model.H, steady_cov_predict_RF), np.transpose(model.H))
    K_steady = np.dot(np.dot(steady_cov_predict_RF, np.transpose(model.H)), np.linalg.inv(S_steady))
    steady_cov_update_RF = np.dot(np.identity(model.dim_x) - np.dot(K_steady, model.H), steady_cov_predict_RF)

    #go back to not rotating RF
    steady_cov_predict = change_reference_frame_rotating(steady_cov_predict_RF, coupling_freq, coupling_phase_shift, t)
    steady_cov_update = change_reference_frame_rotating(steady_cov_update_RF, coupling_freq, coupling_phase_shift, t)

    return steady_cov_predict, steady_cov_update

def change_reference_frame_rotating(object, coupling_freq, coupling_phase_shift, t):
    #TODO define these globally
    R = np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., np.sin(coupling_freq * t + coupling_phase_shift),
                   np.cos(coupling_freq * t + coupling_phase_shift)],
                  [0., 0., -np.sin(coupling_freq * t + coupling_phase_shift),
                   np.cos(coupling_freq * t + coupling_phase_shift)]])
    R_T = R.transpose()
    return np.dot(np.dot(R, object), R_T)