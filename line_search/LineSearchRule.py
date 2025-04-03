import typing as tp
import numpy as np
import math
from scipy.optimize import minimize_scalar

from learning_rate_scheduling import LRScheduler
from gradient import find_gradient


class LineSearchBase(LRScheduler):
    """Базовый класс для методов поиска вдоль направления"""

    def __init__(self,
                 function: tp.Callable[[np.ndarray], float],
                 gradient: tp.Optional[tp.Callable[[np.ndarray], np.ndarray]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.function = function
        self.gradient = gradient
        self._gradient_calculator = None

    def _init_gradient(self):
        """Инициализация градиента если не задан"""
        if self.gradient is None:
            self._gradient_calculator = find_gradient(self.function)
            self.gradient = self._gradient_calculator.get_value

    def _prepare_search(self,
                        x: tp.Optional[np.ndarray],
                        p: tp.Optional[np.ndarray],
                        dimension: int) -> tuple[np.ndarray, np.ndarray]:
        """Общая подготовка для всех методов поиска"""
        if x is None:
            x = np.random.random(dimension)

        if self.gradient is None:
            self._init_gradient()

        if p is None:
            p = -self.gradient(x)
            self.gradient_calls += 1

        return x, p

    def _phi(self, x: np.ndarray, p: np.ndarray) -> tp.Callable[[float], float]:
        """Создание одномерной функции для поиска"""
        return lambda alpha: self.function(x + alpha * p)

    def _update_counters(self, function_calls: int = 0, gradient_calls: int = 0):
        """Обновление счетчиков вызовов"""
        self.function_calls += function_calls
        self.gradient_calls += gradient_calls


class ArmijoRule(LineSearchBase):
    def __init__(self, c1: float = 0.1, beta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.c1 = c1
        self.beta = beta

    def get_lr(self,
               iter_number: tp.Optional[int] = None,
               p: tp.Optional[np.ndarray] = None,
               x: np.ndarray = None,
               **kwargs) -> float:

        x, p = self._prepare_search(x, p, self.dimension)

        alpha = self.initial_lr
        f_x = self.function(x)
        grad = self.gradient(x)
        grad_dot_p = grad.dot(p)

        self._update_counters(function_calls=1, gradient_calls=1)

        while True:
            x_new = x + alpha * p
            f_new = self.function(x_new)
            self._update_counters(function_calls=1)

            if f_new <= f_x + self.c1 * alpha * grad_dot_p:
                return alpha

            alpha *= self.beta
            self.iterations += 1


class GoldsteinRule(LineSearchBase):
    def __init__(self, c1: float = 0.1, c2: float = 0.9, beta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.c1 = c1
        self.c2 = c2
        self.beta = beta

    def get_lr(self,
               iter_number: tp.Optional[int] = None,
               p: tp.Optional[np.ndarray] = None,
               x: np.ndarray = None,
               **kwargs) -> float:

        x, p = self._prepare_search(x, p, self.dimension)

        alpha = self.initial_lr
        f_x = self.function(x)
        grad = self.gradient(x)
        grad_dot_p = grad.dot(p)

        self._update_counters(function_calls=1, gradient_calls=1)

        while True:
            x_new = x + alpha * p
            f_new = self.function(x_new)
            self._update_counters(function_calls=1)

            cond1 = f_new <= f_x + self.c1 * alpha * grad_dot_p
            cond2 = f_new >= f_x + self.c2 * alpha * grad_dot_p

            if cond1 and cond2:
                return alpha
            elif not cond1:
                alpha *= self.beta
            else:
                alpha /= self.beta

            self.iterations += 1


class SciPyLineSearch(LineSearchBase):
    def __init__(self, method: str = 'brent', bounds: tuple = (0, 1), **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.bounds = bounds

    def get_lr(self,
               iter_number: tp.Optional[int] = None,
               p: tp.Optional[np.ndarray] = None,
               x: np.ndarray = None,
               **kwargs) -> float:
        x, p = self._prepare_search(x, p, self.dimension)

        phi = self._phi(x, p)
        result = minimize_scalar(phi, method=self.method)

        self._update_counters(function_calls=result.nfev)
        return result.x


class GoldenSectionSearch(LineSearchBase):
    def __init__(self, bounds: tuple = (0, 1), tol: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.tol = tol
        self.phi_ratio = (math.sqrt(5) - 1) / 2

    def get_lr(self,
               iter_number: tp.Optional[int] = None,
               p: tp.Optional[np.ndarray] = None,
               x: np.ndarray = None,
               **kwargs) -> float:

        x, p = self._prepare_search(x, p, self.dimension)

        phi = self._phi(x, p)
        a, b = self.bounds

        while phi(b) < phi(a):
            a, b = b, 2 * b

        c = b - self.phi_ratio * (b - a)
        d = a + self.phi_ratio * (b - a)
        fc, fd = phi(c), phi(d)

        self._update_counters(function_calls=2)

        while abs(b - a) > self.tol:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - self.phi_ratio * (b - a)
                fc = phi(c)
            else:
                a, c, fc = c, d, fd
                d = a + self.phi_ratio * (b - a)
                fd = phi(d)

            self._update_counters(function_calls=1)
            self.iterations += 1

        return (a + b) / 2