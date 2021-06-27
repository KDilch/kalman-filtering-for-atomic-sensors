#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from atomic_sensor_simulation.state_dynamics_manager.state_dynamics_manager import StateDynamicsManager
from atomic_sensor_simulation.state.atomic_state import AtomicSensorState
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from atomic_sensor_simulation.dynamical_model.atomic_sensor_dynamics import AtomicSensorLinearDifferentialDynamicalModel, AtomicSensorSquareWaveDynamicalModel, AtomicSensorSawtoothDynamicalModel, AtomicSensorSinDynamicalModel


class AtomicStateLinearDynamicsManager(StateDynamicsManager):
    def __init__(self, config, dt):
        #dt must be a parameter to make it possible to reuse this object for a filter
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

        linear_atomic_sensor_dynamics = AtomicSensorLinearDifferentialDynamicalModel(light_correlation_const=config.physical_parameters['light_correlation_const'],
                                                                                     spin_correlation_const=config.physical_parameters['spin_correlation_const'],
                                                                                     larmour_freq=config.physical_parameters['larmour_freq'],
                                                                                     coupling_amplitude=config.coupling['g_p'],
                                                                                     coupling_freq=config.coupling['omega_p'],
                                                                                     coupling_phase_shift=config.coupling['phase_shift'],
                                                                                     intrinsic_noise=config.noise_and_measurement['Q'],
                                                                                     dt=dt)
        StateDynamicsManager.__init__(self,
                                      state_mean=state_mean,
                                      state=state,
                                      dynamics=linear_atomic_sensor_dynamics,
                                      time_step=config.simulation['dt_simulation'],
                                      time=0)


class AtomicStateSquareWaveManager(StateDynamicsManager):
    def __init__(self, config, dt):
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
        square_wave_atomic_sensor_dynamics = AtomicSensorSquareWaveDynamicalModel(light_correlation_const=config.physical_parameters['light_correlation_const'],
                                                                                  spin_correlation_const=config.physical_parameters['spin_correlation_const'],
                                                                                  larmour_freq=config.physical_parameters['larmour_freq'],
                                                                                  coupling_amplitude=config.coupling['g_p'],
                                                                                  coupling_freq=config.coupling['omega_p'],
                                                                                  coupling_phase_shift=config.coupling['phase_shift'],
                                                                                  square_wave_frequency=config.square_waveform['frequency'],
                                                                                  square_wave_amplitude=config.square_waveform['amplitude'],
                                                                                  intrinsic_noise=config.noise_and_measurement['Q'],
                                                                                  dt=dt
                                                                                  )

        StateDynamicsManager.__init__(self,
                                      state_mean=state_mean,
                                      state=state,
                                      dynamics=square_wave_atomic_sensor_dynamics,
                                      time_step=config.simulation['dt_simulation'],
                                      time=0)


class AtomicStateSawtoothWaveManager(StateDynamicsManager):
    def __init__(self, config, dt):
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
        sawtooth_wave_atomic_sensor_dynamics = AtomicSensorSawtoothDynamicalModel(light_correlation_const=config.physical_parameters['light_correlation_const'],
                                                                                  spin_correlation_const=config.physical_parameters['spin_correlation_const'],
                                                                                  larmour_freq=config.physical_parameters['larmour_freq'],
                                                                                  coupling_amplitude=config.coupling['g_p'],
                                                                                  coupling_freq=config.coupling['omega_p'],
                                                                                  coupling_phase_shift=config.coupling['phase_shift'],
                                                                                  sawtooth_wave_frequency=config.sawtooth_waveform['frequency'],
                                                                                  sawtooth_wave_amplitude=config.sawtooth_waveform['amplitude']
                                                                                  )

        StateDynamicsManager.__init__(self,
                                      state_mean=state_mean,
                                      state=state,
                                      dynamics=sawtooth_wave_atomic_sensor_dynamics,
                                      time_step=config.simulation['dt_simulation'],
                                      time=0)


class AtomicStateSinWaveManager(StateDynamicsManager):
    def __init__(self, config, dt):
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
        sin_wave_atomic_sensor_dynamics = AtomicSensorSinDynamicalModel(light_correlation_const=config.physical_parameters['light_correlation_const'],
                                                                        spin_correlation_const=config.physical_parameters['spin_correlation_const'],
                                                                        larmour_freq=config.physical_parameters['larmour_freq'],
                                                                        coupling_amplitude=config.coupling['g_p'],
                                                                        coupling_freq=config.coupling['omega_p'],
                                                                        coupling_phase_shift=config.coupling['phase_shift'],
                                                                        sin_wave_frequency=config.sin_waveform['frequency'],
                                                                        sin_wave_amplitude=config.sin_waveform['amplitude'])

        StateDynamicsManager.__init__(self,
                                      state_mean=state_mean,
                                      state=state,
                                      dynamics=sin_wave_atomic_sensor_dynamics,
                                      time_step=config.simulation['dt_simulation'],
                                      time=0)
