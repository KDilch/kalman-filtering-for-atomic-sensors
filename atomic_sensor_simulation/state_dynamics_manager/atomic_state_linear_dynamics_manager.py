#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from atomic_sensor_simulation.state_dynamics_manager.state_dynamics_manager import StateDynamicsManager
from atomic_sensor_simulation.state.atomic_state import AtomicSensorState
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.dynamical_model.atomic_sensor_dynamics import AtomicSensorLinearDifferentialDynamicalModel


class AtomicStateLinearDynamicsManager(StateDynamicsManager):
    def __init__(self, config):
        state = AtomicSensorState(initial_vec=np.array([config.simulation['spin_y_initial_val'],
                                                        config.simulation['spin_z_initial_val'],
                                                        config.simulation['q_initial_val'],
                                                        config.simulation['p_initial_val']]),
                                  initial_time=0)
        state_mean = AtomicSensorState(initial_vec=np.array([config.simulation['spin_y_initial_val'],
                                                             config.simulation['spin_z_initial_val'],
                                                             config.simulation['q_initial_val'],
                                                             config.simulation['p_initial_val']]),
                                       initial_time=0)
        intrinsic_noise = GaussianWhiteNoise(mean=0.,
                                             cov=config.simulation['R'] / config.simulation['dt_simulation'],
                                             dt=config.simulation['dt_simulation'])
        linear_atomic_sensor_dynamics = AtomicSensorLinearDifferentialDynamicalModel(
            light_correlation_const=config.physical_parameters['light_correlation_const'],
            spin_correlation_const=config.physical_parameters['spin_correlation_const'],
            larmour_freq=config.physical_parameters['larmour_freq'],
            coupling_amplitude=config.coupling['g_p'],
            coupling_freq=config.coupling['omega_p'],
            coupling_phase_shift=config.coupling['phase_shift'])
        StateDynamicsManager.__init__(self,
                                      state_mean=state_mean,
                                      state=state,
                                      intrinsic_noise=intrinsic_noise,
                                      dynamics=linear_atomic_sensor_dynamics,
                                      time_step=config.simulation['dt_simulation'],
                                      time=0)
