import numpy as np
import typing as tp
import gradient_descent as gd
import math as math
if __name__ == "__main__":
    descent = gd.gradient_descent(dimension = 3,
                                  function = lambda point: np.sin(point[0] + point[1] + point[2]),
                                  gradient = lambda point: np.array([math.cos(x) for x in point]),
                                  test_criterion= lambda count: count < 100,
                                  learning_rate_sceduling= lambda count: 1. / count**2)
    print(descent.make_min_value(logging=True))
     