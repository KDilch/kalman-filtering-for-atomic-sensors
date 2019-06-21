import numpy as np
import logging
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class AtomicSensor(object):
    """Implementation of measurement made by a sensor."""
    def __init__(self, state, dt, scalar_strenght_y, logger=None, g_d_COUPLING_CONST=1.):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a AtomicSensor class.')
        self.__state = state
        self.__dt = dt
        self.g_d_COUPLING_CONST = g_d_COUPLING_CONST

        initial_reading = self.g_d_COUPLING_CONST*state.spin + state.quadrature
        self.__noise = GaussianWhiteNoise(initial_reading, scalar_strenght_y, dt)
        self.__z = initial_reading
        self.__z_no_noise = initial_reading
        self.__quadrature_history = []
        self.__quadrature_mean_history = []
        self.__spin_history = []
        self.__spin_mean_history = []
        self.z_no_noise_arr = []

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
    def quadrature_mean_full_history(self):
        return self.__quadrature_mean_history

    @property
    def spin_full_history(self):
        return self.__spin_history

    @property
    def spin_mean_full_history(self):
        return self.__spin_mean_history

    def read(self, t):
        self.__state.step(t)
        self.__z = self.g_d_COUPLING_CONST * self.__state.spin + self.__noise.step()
        self.__z_no_noise = self.g_d_COUPLING_CONST * self.__state.spin
        self.__quadrature_history.append([self.__state.quadrature])
        self.__quadrature_mean_history.append([self.__state.quadrature_mean])
        self.__spin_history.append([self.__state.spin])
        self.__spin_mean_history.append([self.__state.spin_mean])
        self.z_no_noise_arr.append(self.__z_no_noise)
        return self.__z

    def generate(self, num_steps):
        results = np.empty(num_steps)
        for x in range(num_steps):
            results[x] = self.read(x)
        return results
