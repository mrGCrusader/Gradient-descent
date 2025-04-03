import numpy as np

class find_gradient:

    def __init__(self, function):
        self.function = function

    def get_value(self, point: np.array, delta=1e-6) -> np.array:
        print("Gradient_first")
        default_value = self.function(point)
        anw = np.zeros_like(point)

        for i in range(len(point)):
            point[i] += delta
            anw[i] = self.function(point) - default_value
            point[i] -= delta

        return anw / delta
