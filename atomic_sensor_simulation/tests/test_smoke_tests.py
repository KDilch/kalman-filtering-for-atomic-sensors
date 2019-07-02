#!/usr/bin/env python3
import unittest
import logging
import sys
from atomic_sensor_simulation.main import run__atomic_sensor, run_position_speed


class SmokeTest(unittest.TestCase):

    def test_run_smoke_tests(self):
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.error('Running smoke tests...')
        run__atomic_sensor()
        run_position_speed()
