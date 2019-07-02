#!/usr/bin/env python3
import unittest
import numpy as np
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class TestGaussianWhiteNoise(unittest.TestCase):

    def test_noise_values(self):
        val = GaussianWhiteNoise(initial_value=1, scalar_strength=1, dt=1).value
        self.assertIsNotNone(val)

    def test_noise_generate_mean(self):
        epsilon = 0.1
        noise = GaussianWhiteNoise(initial_value=0, scalar_strength=1, dt=1)
        times, noise_arr = noise.generate(100)
        self.assertAlmostEqual(0, np.mean(noise_arr), delta=epsilon)
