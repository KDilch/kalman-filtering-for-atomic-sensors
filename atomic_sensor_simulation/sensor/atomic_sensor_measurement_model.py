# -*- coding: utf-8 -*-
import logging
import numpy as np
from atomic_sensor_simulation.sensor.measurement_model import LinearMeasurementModel
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class AtomicSensorMeasurementModel(LinearMeasurementModel):
    """Implementation of measurement made by an atomic sensor from ICFO paper."""
    def __init__(self, config, logger=None):
        H = np.array([[0., config.noise_and_measurement['gD'], 0., 0.]])
        measurement_noise = GaussianWhiteNoise(mean=0.,
                                               cov=config.simulation['R'] / config.simulation['dt_simulation'],
                                               dt=config.simulation['dt_simulation'])
        LinearMeasurementModel.__init__(self, H, measurement_noise)
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a %s class.' % AtomicSensorMeasurementModel.__name__)