import numpy as np
import logging
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class AtomicSensor(object):
    """Implementation of measurement made by a sensor."""
    def __init__(self, state, dt, scalar_strenght_z, logger=None, g_d_COUPLING_CONST=1.):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a AtomicSensor class.')
        self.__state = state
        self.__dt = dt
        self.g_d_COUPLING_CONST = g_d_COUPLING_CONST

        initial_reading = self.g_d_COUPLING_CONST*state.spin[1] + state.quadrature[1]
        self.__noise = GaussianWhiteNoise(initial_reading, scalar_strenght_z, dt)
        self.__z = initial_reading
        self.__z_no_noise = initial_reading
        self.__state_history = []
        self.__state_mean_history = []
        self.z_no_noise_arr = []

    @property
    def noise(self):
        return self.__noise.value

    @property
    def z_no_noise(self):
        return self.__z_no_noise

    @property
    def state_vec_full_history(self):
        return self.__state_history

    @property
    def state_vec_mean_full_history(self):
        return self.__state_mean_history

    def read(self, t):
        self.__state.step(t)
        self.__z = self.g_d_COUPLING_CONST * self.__state.spin[1] + self.__noise.step()
        self.__z_no_noise = self.g_d_COUPLING_CONST * self.__state.spin[1]

        #append to history
        self.__state_history.append(self.__state.state_vec)
        self.__state_mean_history.append([self.__state.mean_state_vec])
        self.z_no_noise_arr.append(self.__z_no_noise)
        return self.__z

    def generate(self, num_steps):
        results = np.empty(num_steps)
        for x in range(num_steps):
            results[x] = self.read(x)
        return results
