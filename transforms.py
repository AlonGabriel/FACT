import numpy as np


class Normalize:

    def __init__(self, mean, std):
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)

    def __call__(self, array):
        return (array - self.B(self.mean)) / self.B(self.std)

    @classmethod
    def B(cls, stat):  # Broadcast
        return stat[:, np.newaxis, np.newaxis]
