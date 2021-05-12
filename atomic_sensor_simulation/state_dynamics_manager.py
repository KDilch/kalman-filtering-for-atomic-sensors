

class StateDynamicsManager(object):
    def __init__(self, state_mean, state, intrinsic_noise, dynamics, time_step, time):
        self._state_mean = state_mean
        self._state = state
        self._intrinsic_noise = intrinsic_noise
        self._dynamics = dynamics
        self._time_step = time_step
        self._time = time

    def step(self, time):
        self._time = time
        self._dynamics.step(self._state_mean, self._state, self._time, self._time_step, self._intrinsic_noise)

