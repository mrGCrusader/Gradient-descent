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
    def __init__(self, func, start, file_name, maxIter = 12000, gradient = None):
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


        plt.plot(self.iters, z, color=mode, marker='.', label=name)
        # plt.scatter(self.iters, z, color=mode, marker='.', linestyle=None)
        # plt.scatter(self.iters, z, c=mode, marker='.')
        # host.scatter(iters, z, color=mode, marker='.')
        print(f"{name}: {anw}, {descent.get_logs()[0]}")

    def addAll(self, steps):
        for x in steps: self.add(*x)
if __name__=='__main__':
    ex = Example(lambda x: 0.1 * x[0]**2 + 2 * x[1]**2, np.array([10., 10.]), "some_name")
    # ex = Example(lambda x: x[0]**2 + x[1]**2, np.array([10., 10.]), "some_name")
    ex.addAll([
        # [ lrs.Constant(0.9), "const 0.9", 'k' ],
        # [ lrs.Constant(0.1), "const 0.1", 'c' ],
        # [ lrs.Constant(0.01), "const 0.01", 'r' ],
        # [ lrs.Constant(0.001), "const 0.001", 'y' ],
        # [ lrs.Constant(0.0001), "const 0.0001", 'k' ]
        # [ lrs.ExponentialDecay(0.7, 1), "exp", 'b' ],
        # [ lrs.ExponentialDecay(0.2, 0.05), "exp", 'r' ],#good for harder
        # [ lrs.ExponentialDecay(0.5, 0.005), "exp", 'k' ],#good for x^2+y^2
        # [ lrs.ExponentialDecay(0.2, 0.1), "exp ", 'y' ]
        # [ lrs.PolynomialDecay(), "polynomial", 'b' ],
        [ lrs.PolynomialDecay(), "polynomial", 'b' ],
        # [ lrs.ExponentialDecay(0.2, 0.05), "exp", 'r' ],
        # [ lrs.ExponentialDecay(0.2, 0.1), "exp", 'y' ]
        # [ lrs.TimeBasedDecay(), "inv", 'k' ],
    ])
    plt.legend(loc="upper right")
    plt.show()
