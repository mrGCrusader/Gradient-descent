import matplotlib.pyplot as plt
import numpy as np

import gradient_descent as gd
import learning_rate_scheduling as lrs

number = 2


class Example:

    def __painting_3d(self, x, y, z, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x, y, z, c='r', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Траектория градиентного спуска')

        plt.show()

        plt.savefig(file_name)

    def run_example(self,
                    dimension=2,
                    function=(lambda point: 20 * point[0] ** 2 + 20 * point[1] ** 2),
                    gradient=None,
                    test_criterion=lambda count: count < 1000,
                    learning_rate_scheduling=lrs.PolynomialDecay(),
                    file_name='/home/crusader/ml_yandex/Gradient-descent/graphics/first_ex.png'):
        descent = gd.gradient_descent(dimension, function, gradient, test_criterion, learning_rate_scheduling)
        descent.make_min_value()
        lst = [point for (_, point) in descent.get_log()]
        x = [point[0] for point in lst]
        y = [point[1] for point in lst]
        z = [function(point) for point in lst]
        self.__painting_3d(x, y, z, file_name)

YOUR_FILE_NAME = '/home/crusader/ml_yandex/Gradient-descent/graphics/second_ex.png'

if __name__ == "__main__":
    ex = Example()
    ex.run_example(file_name=YOUR_FILE_NAME)
    number += 1
