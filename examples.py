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

        # Показать график
        plt.show()

        # Сохранение графика в файл (с правильным расширением)
        plt.savefig(file_name)

    def run_example(self,
                    dimension=2,
                    function=(lambda point: 20 * point[0] ** 2 - 20 * point[1] ** 2),
                    gradient=lambda point: np.array([40 * x for x in point]),
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


if __name__ == "__main__":
    ex = Example()
    ex.run_example(file_name=f'C:/Users/ayazm/PycharmProjects/Gradient-descent/graphics/{number}_ex.png')
    number += 1
