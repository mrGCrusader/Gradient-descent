import os

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import line_search as ls
from learning_rate_scheduling import LRScheduler, Constant, StepDecay

# from line_search import GoldenSectionSearch, ArmijoRule, GoldsteinRule, SciPyLineSearch

import math
import gradient_descent as gd
import learning_rate_scheduling as lrs
import random
import stop_criteria as sc
import plotly.graph_objects as go

YOUR_DIR_NAME = os.path.abspath('../ex_graphics/')

class Example:
    def __init__(self, func, start, file_name, maxIter = 30000, gradient = None):
        self.function = func
        self.gradient = gradient
        self.start = start
        self.maxIter = maxIter
        self.file_name = file_name
        plt.xlabel("Iterations")
        plt.ylabel("z min")
        self.iters = list(range(maxIter))

    def add(self, step, name, mode):
        descent = gd.gradient_descent(2, self.function, self.gradient, sc.Combine(stop1=sc.MaxIterations(self.maxIter), stop2=sc.Convergence()), step)
        descent.make_min_value(self.start)
        lst = [point for (_, point) in descent.get_enum_point()]
        x = [point[0] for point in lst]
        y = [point[1] for point in lst]
        z = [self.function(point) for point in lst]
        anw = z[-1]
        while len(z) < self.maxIter:
            z.append(-1)
        plt.scatter(self.iters, z, c=mode)
        # host.scatter(iters, z, color=mode, marker='.')
        print(f"{name}: {anw}, {descent.get_logs()[0]}")

    def addAll(self, steps):
        for x in steps: self.add(*x)
if __name__=='__main__':
    ex = Example(lambda x: 10 * x[0]**2 + x[1]**2, np.array([10., 10.]), "some_name")
    ex.addAll([
        [ lrs.Constant(0.1), "const", 'b' ],
        [ lrs.Constant(0.01), "const", 'r' ],
        [ lrs.Constant(0.001), "const", 'y' ],
        [ lrs.Constant(0.0001), "const", 'k' ]
        # [ lrs.ExponentialDecay(0.7, 1), "const", 'b.' ],
        # [ lrs.ExponentialDecay(0.2, 0.05), "const", 'r' ],
        # [ lrs.ExponentialDecay(0.2, 0.1), "const", 'y.' ]
        # [ lrs.PolynomialDecay(), "const", 'b' ],
        # [ lrs.PolynomialDecay(), "const", 'b.' ],
        # [ lrs.ExponentialDecay(0.2, 0.05), "const", 'r.' ],
        # [ lrs.ExponentialDecay(0.2, 0.1), "const", 'y.' ]
    ])
    plt.show()
