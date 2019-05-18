import numpy as np
import logging
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class SpeedVelocitySensor(object):
    """Implementation of measurement made by a sensor."""
    def __init__(self, state, dt, scalar_strenght_y, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a AtomicSensor class.')
        self.__state = state
        self.__dt = dt
        self.__noise = GaussianWhiteNoise(initial_reading, scalar_strenght_y, dt)
        self.__z = 0  # photocurrent value with noise (measured by the atomic sensor)
        self.__z_no_noise = initial_reading  # photocurrent value without noise
        self.__quadrature_history = []
        self.__quadrature_no_noise_history = []

    @property
    def noise(self):
        return self.__noise.value

    @property
    def z_no_noise(self):
        return self.__z_no_noise

    @property
    def quadrature_full_history(self):
        return self.__quadrature_history

    @property
    def quadrature_no_noise_full_history(self):
        return self.__quadrature_no_noise_history

    def read(self, t):
        self.__state.step(t)
        self.__z = self.__z_no_noise + g_d_COUPLING_CONST * self.__state.spin * self.__dt + self.__noise.step()
        self.__z_no_noise += g_d_COUPLING_CONST * self.__state.spin_no_noise * self.__dt
        self.__quadrature_history.append([self.__state.quadrature])
        self.__quadrature_no_noise_history.append([self.__state.quadrature_no_noise])

        return self.__z

    def generate(self, num_steps):
        results = np.empty(num_steps)
        for x in range(num_steps):
            results[x] = self.read(x)
        return results
