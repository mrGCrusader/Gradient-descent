import numpy as np
import scipy as sc

class find_gradient_sc:
    def __init__(self, function, delta=1e-6):
        self.function = function
        self.delta = delta

    def find_gradient(self, point: np.array) -> np.array:
        return sc.optimize.approx_fprime(point, self.function, self.delta)
