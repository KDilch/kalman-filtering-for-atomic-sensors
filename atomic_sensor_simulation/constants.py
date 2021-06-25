#!/usr/bin/env python
# -*- coding: utf-8 -*-
from atomic_sensor_simulation.singleton import Singleton


class AtomicSensorConstants(metaclass=Singleton):

    def __init__(self):
        self.__ATOMIC_SENSOR_DYNAMICS_TYPES = ['linear', 'sin', 'square', 'sawtooth']
        self.__ATOMIC_SENSOR_FILTER_TYPES = ['lkf', 'ukf', 'ekf']

    @property
    def ATOMIC_SENSOR_DYNAMICS_TYPES(self):
        return self.__ATOMIC_SENSOR_DYNAMICS_TYPES

    @property
    def ATOMIC_SENSOR_FILTER_TYPES(self):
        return self.__ATOMIC_SENSOR_FILTER_TYPES
