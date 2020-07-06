#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC
import logging


class Model(ABC):
    """An abstract class representing any filter_model (noise+measurement+process)."""

    def __init__(self,
                 Q,
                 R,
                 R_delta,
                 Gamma,
                 u,
                 z0,
                 dt,
                 logger=None):
        self._logger = logger or logging.getLogger(__name__)
        self.dt = dt
        self.Q = Q
        self.R_delta = R_delta
        self.R = R
        self.dim_z = len(z0)
        self.u_control_vec = u
        self.Gamma_control_transition_matrix = Gamma
