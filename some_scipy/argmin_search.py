from scipy.optimize import minimize_scalar

from learning_rate_scheduling import LRScheduler


class ArgminSearcherScipy(LRScheduler):
    def __init__(self, function, method='brent'):
        super().__init__()
        self.function = function
        self.method = method


    def get_lr(self, iter_number=None, **kwargs):
        return minimize_scalar(
            fun=self.function,
            method=self.method,
        )

first =  ArgminSearcherScipy(function=lambda x: x**2)
# print(first.find_argmin().x)
# result = first.find_argmin()
# print("Оптимальное значение x:", result.x)
# print("Минимальное значение функции:", result.fun)
# print("Количество итераций:", result.nfev)
