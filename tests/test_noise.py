#!/usr/bin/env python3
import unittest
from atomic_sensor_simulation.noise import GaussianWhiteNoise


class TestGaussianWhiteNoise(unittest.TestCase):

    def test_values(self):
        time, vals = GaussianWhiteNoise(initial_value=1,scalar_strength=1,dt=1).values
        self.assertIsNotNone(time)
        self.assertIsNotNone(vals)
