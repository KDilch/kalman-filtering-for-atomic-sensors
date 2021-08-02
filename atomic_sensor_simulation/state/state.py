# -*- coding: utf-8 -*-
from abc import ABC
import logging


class State(ABC):
    """An abstract class representing any dynamical_model vector."""

    def __init__(self,
                 initial_vec,
                 initial_time,
                 logger=None):

        self._logger = logger or logging.getLogger(__name__)
        self._vec = initial_vec
        self._time = initial_time

    @property
    def vec(self):
        return self._vec

    @property
    def time(self):
        return self._time

    def update(self, value):
        if self._vec.shape == value.shape:
            self._vec = value
        else:
            raise ValueError(f"The shape of the vector is incorrect. It is {value.shape} and should be {self._vec.shape}")
