import numpy as np
import math
class find_gradient:

    def __init__(self, function):
        self.function = function

    def get_value(self, point: np.array, delta=1e-6) -> np.array:
        default_value = self.function(point)
        anw = np.zeros_like(point)

        for i in range(len(point)):
            point[i] += delta
            print(default_value)
            print(self.function(point))
            print(self.function(point) - default_value)
            anw[i] = self.function(point) - default_value
            point[i] -= delta
        return anw / delta

    def __find_norm(point: np.array) -> np.array:
        square = point * point
        return math.sqrt(square.sum())
    
    def get_modifier_value(self, point: np.array, delta=1e-6, upper_bound=4.) -> np.array:
        default_value = self.function(point)
        anw = np.zeros_like(point)

        for i in range(len(point)):
            point[i] += delta
            anw[i] = self.function(point) - default_value
            point[i] -= delta
        grad = anw / delta
        while (self.__find_norm(grad) > upper_bound):
            grad = np.random.random(len(point))
        return grad
