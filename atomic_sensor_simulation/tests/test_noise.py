#!/usr/bin/env python3
import unittest
import numpy as np
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class TestGaussianWhiteNoise(unittest.TestCase):

    def test_noise_values(self):
        noise = GaussianWhiteNoise(mean=1, cov=1, dt=1)
        noise.step()
        self.assertIsNotNone(noise.value)

    def test_noise_generate_mean(self):
        epsilon = 0.1
        noise = GaussianWhiteNoise(mean=0, cov=0.1, dt=1)
        times, noise_arr = noise.generate(100)
        self.assertAlmostEqual(0, np.sum(noise_arr)/len(noise_arr), delta=epsilon)
