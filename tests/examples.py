
import os

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import gradient_descent as gd  # Теперь импорт работает
import learning_rate_scheduling as lrs
import stop_criteria as sc
import plotly.graph_objects as go

YOUR_DIR_NAME = os.path.abspath('../ex_graphics/')


class Example:

    # def __painting_3d_with_mathplotlib(self, x, y, z, file_name):
    #     if not os.path.exists(file_name):
    #         with open(file_name, "wb") as f:
    #             pass

    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')

    #     anw = [x[len(x) - 1], y[len(y) - 1], z[len(z) - 1]]
    #     ax.scatter(anw[0], anw[1], anw[2], c='b', marker='o')
    #     ax.scatter(x, y, z, c='r', marker='o')

    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Траектория градиентного спуска')
    #     # plt.show()
    #     # plt.show()
    #     plt.savefig(file_name)
        
    def __painting_3d_with_plotly(self, x, y, z, file_name, function):
        row = np.linspace(max(min(x), -50), min(max(x), 50), 1)  # 50 точек вместо случайных
        col = np.linspace(max(min(y), -50), min(max(y), 50), 1)
        X, Y = np.meshgrid(row, col)

        Z = np.array([[function((x_i, y_j)) for x_i in row] for y_j in col])
        
        # поверхность
        surface = go.Surface(x = X, y = Y, z = Z,
                             opacity=0.7,
                             contours_z={'show': True})        
        # print(len(x), len(y), len(z))
        # >> 135 135 135    
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
                    learning_rate_scheduling=lrs.Constant(),
                    file_name='/home/crusader/ml_yandex/Gradient-descent/graphics/first_ex.png',
                    beginning_point=None) -> list:
        descent = gd.gradient_descent(dimension, function, gradient, test_criterion, learning_rate_scheduling)
        descent.make_min_value(beginning_point)
        lst = [point for (_, point) in descent.get_log()]
        x = [point[0] for point in lst] 
        y = [point[1] for point in lst]
        z = [function(point) for point in lst]
        anw = [x[len(x) - 1], y[len(y) - 1], z[len(z) - 1]]
        self.__painting_3d_with_plotly(x, y, z, file_name, function)
        return anw


class Generate_test:

    def __init__(self, count: int):
        self.count = count
        self.ex = Example()

    def generate(self):
        for num in range(self.count):
            [alpha, bravo, charlie] = np.random.randint(-10, 10, size=3)
            self.ex.run_example(dimension=2,
                                function=(
                                    lambda point: alpha * point[0] ** 2 + bravo * point[1] ** 2 + charlie * point[1] *
                                                  point[0]),
                                gradient=None,
                                file_name=f'{YOUR_DIR_NAME}/{num}.png'
                                )

def create_dir():
    if not os.path.isdir(YOUR_DIR_NAME):
        if YOUR_DIR_NAME not in sys.path:
            os.makedirs(YOUR_DIR_NAME)

if __name__ == "__main__":
    create_dir()
    generator = Generate_test(1)
    generator.generate()
