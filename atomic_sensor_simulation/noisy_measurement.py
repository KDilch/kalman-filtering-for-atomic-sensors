# -*- coding: utf-8 -*-


class NoisyMeasurementGenerator(object):

    def __init__(self, signal, noise='gaussian'):
        self.__val = None