from scipy.optimize import minimize_scalar

class ArgminSearcherScipy:
    def __init__(self, function, method='brent'):
        self.function = function
        self.method = method


    def find_argmin(self):
        return minimize_scalar(
            fun=self.function,
            method=self.method,
        )

first =  ArgminSearcherScipy(function=lambda x: x**2)
# print(first.find_argmin().x)
result = first.find_argmin()
print("Оптимальное значение x:", result.x)
print("Минимальное значение функции:", result.fun)
print("Количество итераций:", result.nfev)
