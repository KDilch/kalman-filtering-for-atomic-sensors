from abc import ABC


class HistoryManager(ABC):
    def __init__(self):
        return

    def add_history_point(self, history_point):
        raise NotImplementedError('This is an interface method, please provide a specific implementation.')