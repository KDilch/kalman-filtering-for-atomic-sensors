import numpy as np
import logging
from atomic_sensor_simulation.noise import GaussianWhiteNoise
from numpy.random import randn



class PosSensor(object):
    """Implementation of measurement made by a sensor."""
    def __init__(self, state, dt, scalar_strenght_y, logger=None):
        self.__logger = logger or logging.getLogger(__name__)
        self.__logger.info('Initializing an instance of a AtomicSensor class.')
        self.__state = state
        self.__dt = dt
        initial_reading = np.array([state.position_x, state.position_y])
        self.__noise = GaussianWhiteNoise(initial_reading, scalar_strenght_y, dt)
        self.__z = np.transpose(initial_reading)
        self.__z_no_noise = np.transpose(initial_reading)
        self.__position_x_history = []
        self.__position_x_no_noise_history = []
        self.__noise_std = scalar_strenght_y

    @property
    def noise(self):
        return self.__noise.value

    @property
    def z_no_noise(self):
        return self.__z_no_noise

    @property
    def pos_x_full_history(self):
        return self.__position_x_history

    @property
    def pos_x_no_noise_full_history(self):#TODO
        return self.__position_x_no_noise_history

    def read(self, t):
        #TODO redo using H
        self.__state.step(t)
        self.__z = [self.__z_no_noise[0] + self.__state.velocity_x * self.__dt + randn() * self.__noise_std,
                    self.__z_no_noise[1] + self.__state.velocity_y * self.__dt + randn() * self.__noise_std]
        self.__z_no_noise = np.array([self.__z_no_noise[0] + self.__state.velocity_x * self.__dt,
                                      self.__z_no_noise[1] + self.__state.velocity_y * self.__dt])
        self.__position_x_history.append([self.__state.position_x])
        self.__position_x_no_noise_history.append([self.__state.mean_position_x])
        return self.__z

    def generate(self, num_steps):
        results = np.empty(num_steps)
        for x in range(num_steps):
            results[x] = self.read(x)
        return results
