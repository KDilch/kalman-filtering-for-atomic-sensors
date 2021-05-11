

class StateDynamicsManager(object):
    def __init__(self, state_mean, state, intrinsic_noise, dynamics):
        self._state_mean = state_mean
        self._state = state
        self._intrinsic_noise = intrinsic_noise
        self._dynamics = dynamics

