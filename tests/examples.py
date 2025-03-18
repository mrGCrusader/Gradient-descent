import os

import matplotlib.pyplot as plt
import numpy as np

import gradient_descent as gd
import learning_rate_scheduling as lrs
import stop_criteria as sc
from scipy.optimize import root
from gradient_descent import Find_gradient


class Example:
    def __painting_3d(self, x, y, z, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x, y, z, c='r', marker='o')
        # ax.scatter(*extremum, c = 'b', marker = 'o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Траектория градиентного спуска')

        plt.show()

        plt.savefig(file_name)

    def __find_extremum(self, function, gradient):
        initial_guess = np.array([1.0, 1.0])
        result = root(gradient, initial_guess)
        return np.array(result)

    def run_example(self,
                    dimension=2,
                    function=(lambda point: 20 * point[0] ** 2 + 20 * point[1] ** 2),
                    gradient=None,
                    test_criterion=sc.Convergence(),
                    learning_rate_scheduling=lrs.PolynomialDecay(),
                    file_name='/home/crusader/ml_yandex/Gradient-descent/graphics/first_ex.png'):
        descent = gd.gradient_descent(dimension, function, gradient, test_criterion, learning_rate_scheduling)
        descent.make_min_value()
        lst = [point for (_, point) in descent.get_log()]
        x = [point[0] for point in lst]
        y = [point[1] for point in lst]
        z = [function(point) for point in lst]
        self.__painting_3d(x, y, z, file_name)


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
                                file_name=f'/home/crusader/ml_yandex/Gradient-descent/graphics/{num}_ex.png')


YOUR_FILE_NAME = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    generator = Generate_test(10)
    generator.generate()
