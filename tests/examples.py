import os

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import gradient_descent as gd
import learning_rate_scheduling as lrs
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
        
        # управляет изображением
        fig = go.Figure(data=[scatter, surface])
        fig.show()

    def run_example(self,
                    dimension=2,
                    function=(lambda point: 20 * point[0] ** 2 + 20 * point[1] ** 2),
                    gradient=None,
                    test_criterion=sc.Convergence(),
                    learning_rate_scheduling: lrs.LRScheduler() = lrs.Constant(),
                    file_name='/home/crusader/ml_yandex/Gradient-descent/graphics/first_ex.png',
                    beginning_point=None) -> list:
        descent = gd.gradient_descent(dimension, function, gradient, test_criterion, learning_rate_scheduling)
        descent.make_min_value(beginning_point)
        lst = [point for (_, point) in descent.get_enum_point()]
        x = [point[0] for point in lst] 
        y = [point[1] for point in lst]
        z = [function(point) for point in lst]
        anw = [x[len(x) - 1], y[len(y) - 1], z[len(z) - 1]]
        self.__painting_3d_with_plotly(x, y, z, file_name, function)
        print(f"iterations, function_calls, gradient_calls: {descent.get_logs()}")
        return anw


class Generate_test:

    def __init__(self, count: int):
        self.count = count
        self.ex = Example()

    def generate(self):
        for num in range(self.count):
            [alpha, bravo, charlie] = np.random.randint(-10, 10, size=3)
            # alpha, bravo, charlie = 1, 4, 3
            fun_gen = (lambda point: alpha * point[0] ** 2 + bravo * point[1] ** 2 + charlie * point[1] *
                                                  point[0])
            self.ex.run_example(dimension=2,
                                function=fun_gen,
                                gradient=None,
                                file_name=f'{YOUR_DIR_NAME}/{num}.png',
                                learning_rate_scheduling=lrs.ArmijoRule(
                                    function=fun_gen),
                                )

def create_dir():
    if not os.path.isdir(YOUR_DIR_NAME):
        if YOUR_DIR_NAME not in sys.path:
            os.makedirs(YOUR_DIR_NAME)

def first_ex(ex: Example):
    
    ex.run_example(dimension=2,
                   function=lambda x: x[0]**2 + x[1]**2,
                   gradient=None,
                   beginning_point=np.array([10., 10.]))
    
    
def second_ex(ex: Example):
    ex.run_example( dimension= 2,
        function= lambda x: max(x[0]**2 + x[1] - 2, x[0]**2 - x[1] - 2) + 10,
        gradient=None,
        beginning_point=np.array([0., 0.])
    )
if __name__ == "__main__":
    # create_dir()
    # generator = Generate_test(1)
    # generator.generate()
    ex = Example()
    # first_ex(ex)
    second_ex(ex)
    
    

