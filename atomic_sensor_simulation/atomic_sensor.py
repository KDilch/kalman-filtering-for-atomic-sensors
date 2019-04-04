import numpy as np
import logging
from atomic_sensor_simulation.CONSTANTS import g_d_COUPLING_CONST
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class AtomicSensor(object):
    """Implementation of measurement made by a sensor."""
    def __init__(self, state, dt, scalar_strenght_y, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a AtomicSensor class.')
        self.__state = state
        self.__dt = dt
        from atomic_sensor_simulation.CONSTANTS import g_d_COUPLING_CONST
        initial_reading = g_d_COUPLING_CONST*state.spin + state.quadrature
        self.__noise = GaussianWhiteNoise(initial_reading, scalar_strenght_y, dt)
        self.__z = initial_reading  # photocurrent value with noise (measured by the atomic sensor)
        self.__z_no_noise = initial_reading  # photocurrent value without noise

    @property
    def noise(self):
        return self.__noise.value

    @property
    def z_no_noise(self):
        return self.__z_no_noise

    def read(self,t):
        self.__state.step(t)
        self.__z = self.__z_no_noise + g_d_COUPLING_CONST * self.__state.spin * self.__dt + self.__noise.step()
        self.__z_no_noise += g_d_COUPLING_CONST * self.__state.spin_no_noise * self.__dt
        return self.__z, self.__state.quadrature

    def generate(self, num_steps):
        results = np.empty(num_steps)
        for x in range(num_steps):
            results[x] = self.read()
        return results
