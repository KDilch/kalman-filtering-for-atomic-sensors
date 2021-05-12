# -*- coding: utf-8 -*-
from state.state import State


class AtomicSensorState(State):
    """
    Specialization of a dynamical_model abstract class. Represents a spin-quadrature state vector x_t_k = [j, q].
    """

    def __init__(self, initial_vec, initial_time, logger=None):
        """
        :param initial_vec: ndarray
        :param initial_time: float
        :param logger: an instance of logger.Logger; if not passed a new instance of a logger is initialized
        """
        State.__init__(self,
                       initial_vec=initial_vec,
                       initial_time=initial_time,
                       logger=logger)

    @property
    def spin_vec(self):
        return [self._vec[0], self._vec[1]]

    @property
    def quadrature_vec(self):
        return [self._vec[2],
                self._vec[3]]

    @property
    def spin_y(self):
        return self._vec[0]

    @property
    def spin_z(self):
        return self._vec[1]

    @property
    def quadrature_p(self):
        return self._vec[2]

    @property
    def quadrature_q(self):
        return self._vec[3]
