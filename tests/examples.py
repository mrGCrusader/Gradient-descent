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

    def __painting_3d_with_plotly(self, x, y, z, file_name, function):
        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)

        x_padding, y_padding = (x_max - x_min) * 0.5, (y_max - y_min) * 0.5

        row = np.linspace(x_min - x_padding, x_max + x_padding, 50)
        col = np.linspace(y_min - y_padding, y_max + y_padding, 50)
        X, Y = np.meshgrid(row, col)

        Z = np.array([[function((x_i, y_j)) for x_i in row] for y_j in col])

        # поверхность
        surface = go.Surface(x = X, y = Y, z = Z,
                             opacity=0.7,
                             contours_z={'show': True})

        # добавить линию
        scatter = go.Scatter3d(x=x, y=y, z=z,
                               mode='lines+markers',
                               line=dict(color='red', width=5),
                               marker=dict(size=3)
                               )

        # добавляем последнюю точку
        last_point = go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]],
                                  mode='markers',
                                  marker=dict(color='green', size=6)
                                  )
        # управляет изображением
        fig = go.Figure(data=[scatter, surface, last_point])
        fig.show()

    def run_example(self,
                    dimension=2,
                    function=(lambda point: 20 * point[0] ** 2 + 20 * point[1] ** 2),
                    gradient=None,
                    test_criterion: sc.StoppingCriterion =sc.Convergence(),
                    learning_rate_scheduling: LRScheduler = Constant(),
                    file_name='/home/crusader/ml_yandex/Gradient-descent/graphics/first_ex.png',
                    beginning_point=None) -> list:
        descent = gd.gradient_descent(dimension, function, gradient, test_criterion, learning_rate_scheduling)
        descent.make_min_value(beginning_point)
        lst = [point for (_, point) in descent.get_enum_point()]
        x = [point[0] for point in lst]
        y = [point[1] for point in lst]
        z = [function(point) for point in lst]
        anw = [x[-1], y[-1], z[-1]]
        self.__painting_3d_with_plotly(x, y, z, file_name, function)
        print(f"iterations, function_calls, gradient_calls: {descent.get_logs()}")
        return anw


class Generate_test:

    def __init__(self, count: int):
        self.count = count
        self.ex = Example()

    def generate(self):
        for num in range(self.count):
            [alpha, bravo, charlie] = np.random.randint(1, 10, size=3)
            fun_gen = (lambda point: alpha * point[0] ** 2 + bravo * point[1] ** 2 + charlie * point[1] *
                                     point[0])
            self.ex.run_example(dimension=2,
                                function=fun_gen,
                                gradient=None,
                                file_name=f'{YOUR_DIR_NAME}/{num}.png',
                                learning_rate_scheduling=StepDecay()
                                )

def create_dir():
    if not os.path.isdir(YOUR_DIR_NAME):
        if YOUR_DIR_NAME not in sys.path:
            os.makedirs(YOUR_DIR_NAME)

def ex_sample(func, test_criterion = sc.Convergence(), gradient = None, beginning_point = np.array([10., 10.])):
    for i in (lrs.Constant(), lrs.TimeBasedDecay(), lrs.ExponentialDecay()):
        anw = ex.run_example(dimension=2,
                             function=func,
                             gradient=gradient,
                             learning_rate_scheduling=i,
                             test_criterion=test_criterion,
                             beginning_point=beginning_point)[0]
        print(anw)
        print(f'anw = {anw}')

def first_ex():
    func = lambda x: x[0]**2 + x[1]**2
    ex_sample(func, sc.Convergence())


def second_ex():
    """
    bad function for this graphic
    """
    func = lambda x: max(x[0]**2 + x[1] - 2, x[0]**2 - x[1] - 2) + 10
    ex_sample(func, sc.MaxIterations(200))

def third_ex():
    """
    very bad function without min value
    """
    func = lambda x:  x[0]**2 + x[1] - 1
    ex_sample(func, sc.MaxIterations(200))

def fourth_ex():
    """
    strange_function
    """
    func = lambda x: 100 * math.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10)
    ex_sample(func, sc.Combine(sc.MaxIterations(10000), sc.Convergence()))

def eight_ex():
    """
    noisy function
    """
    func = lambda x: x[0]**2 + x[1]**2 + random.random()
    ex_sample(func, sc.MaxIterations(200))



def ex_sample1(func, str_decay, it_count = 300, gradient=None, beginning_point=np.array([10., 10.])):
    mp2 = {
        "time" : lrs.TimeBasedDecay(),
        "exp" : lrs.ExponentialDecay(),
        "const" : lrs.Constant(),
        "poly"  : lrs.PolynomialDecay(),

        "armijo" : ls.ArmijoRule(function=func),
        "goldstein" : ls.GoldsteinRule(function=func),
        "scipy" :  ls.SciPyLineSearch(function=func),
        "goldensection" :  ls.GoldenSectionSearch(function=func),
    }
    anw = ex.run_example(dimension=2,
                         function=func,
                         gradient=gradient,
                         learning_rate_scheduling=mp2[str_decay],
                         test_criterion=sc.MaxIterations(it_count),
                         beginning_point=beginning_point)[2]
    print(str_decay, anw)

def run(func, counts):
    for x in counts:
        ex_sample1(func, "time", it_count=x)
        ex_sample1(func, "exp", it_count=x)
        ex_sample1(func, "const", it_count=x)
        ex_sample1(func, "poly", it_count=x)
def run_hard(func, counts):
    for x in counts:
        ex_sample1(func, "scipy", it_count=x)
        ex_sample1(func, "armijo", it_count=x)
        ex_sample1(func, "goldstein", it_count=x)
def nine_ex():
    """
    one more
    """
    func = lambda x: 0.1 * x[0]**2 + 2 * x[1]**2
    ex_sample(func, sc.Convergence())


if __name__ == "__main__":
    ex = Example()
    # run(lambda x: 100 * math.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10), [1000, 10000])
    run_hard(lambda x: 100 * math.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10), [300, 1000])
    # run1_hard()
    # ex_sample1(lambda x : x[0] ** 2 + 2, "goldstein", it_count=1000)
    # ex_sample1(lambda x : x[0] ** 2 + 2, "armijo", it_count=1000)
    # ex_sample1(lambda x : x[0] ** 2 + 2, "scipy", it_count=1000)
    # run1_hard()
    # nine_ex()
    # run2()
    # ex_sample1(lambda x : x[0]**2 + 1, "scipy", it_count=100000)
    # run2_hard()
    # ex_sample1("armijo", "iter10000")
    # eight_ex()


