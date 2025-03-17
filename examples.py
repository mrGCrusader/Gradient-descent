import numpy as np
import typing as tp
import gradient_descent as gd
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def printing_3d(x, y, z, file_name):
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
   
   
def printing_2d(x, y, file_name):
    plt.figure(figsize=(8, 6))  # Размер графика
    plt.scatter(x, y, c='r', marker='o')  # Точечный график
    plt.plot(x, y)  # Линия, соединяющая точки
    plt.xlim([min(x) - 1, max(x) + 1])  # Масштаб по оси X
    plt.ylim([min(y) - 1, max(y) + 1])  # Масштаб по оси Y


    plt.show() 
    plt.savefig(file_name)
if __name__ == "__main__":
    descent = gd.gradient_descent(dimension = 2,
                                  function = (function:=lambda point: 20 * point[0]**2 - 20 * point[1]**2),
                                  gradient = lambda point: np.array([40 * x for x in point]),
                                  test_criterion= lambda count: count < 1000,
                                  learning_rate_sceduling = lambda count: min(1 / count**2, 0.01))
    print(descent.make_min_value())
    lst = [point for (_, point) in descent.get_log()]
    x = [point[0] for point in lst]
    y = [point[1] for point in lst]
    z = [function(point) for point in lst]
    # print(lst)
    # for i in x:
    #     print(i)
    # print("------------------")
    # for i in y:
    #     print(i)
    # print(x, y, z)
    # printing_2d(x, y, '/home/crusader/ml_yandex/Gradient-descent/graphics/first_ex.png')
    printing_3d(x, y, z, '/home/crusader/ml_yandex/Gradient-descent/graphics/first_ex.png')