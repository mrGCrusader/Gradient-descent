import typing as tp

import numpy as np

import stop_criteria as sc
from gradient import find_gradient
from learning_rate_scheduling import LRScheduler


class ArmijoRule(LRScheduler):
    def __init__(self, dimension: int = 2,
                 function: tp.Callable[[np.array], float] = lambda arg: np.sum(np.sin(arg)),
                 gradient: tp.Callable[[np.array], np.array] = None,
                 test_criterion: sc.StoppingCriterion = sc.Convergence(),
                 alpha_0: float = 1):
        """
        Armijo rule for choosing learning rate.

        Parameters:
        - function: the function that we minimize
        - gradient: the gradient of the function f
        - p: direction of descent ---- нужно ли??? ----
        - alpha_0: initial step (default 1.0)
        - beta: step reduction factor (default 0.5)
        - c1: Armijo condition parameter (default 0.1)

        return:
        - alpha: chosen learning rate
        """
        super().__init__(alpha_0)
        self.dimension = dimension
        self.function = function
        self.gradient = gradient
        self.test = test_criterion
        self.alpha = alpha_0

    def get_lr(self, iter_number: int = 1,
                       x: np.array = None,
                       p: np.array = None,
                       beta=0.5, c1=0.1):

        alpha = self.alpha
        fun_value_x = self.function(x)
        if x is None:
            x = np.random.random(self.dimension)
        if self.gradient is None:
            find_grad = find_gradient(self.function)
            self.gradient = find_grad.get_value
        grad_x = self.gradient(x)
        if p is None:
            p = -grad_x

        step_number: int = 1
        curr_value: np.typing.NDArray = x.copy()
        grad_dot_p = grad_x.dot(p)

        if grad_dot_p >= 0:
            p = -grad_x
            grad_dot_p = grad_x.dot(p)

        while not self.test.should_stop(step=curr_value,
                                        value=self.function(curr_value),
                                        gradient=self.gradient,
                                        iteration=step_number):
            curr_value = curr_value + alpha * p
            fx_new = self.function(curr_value)
            armijo_condition = fx_new <= fun_value_x + c1 * alpha * grad_dot_p

            if armijo_condition:
                return alpha
            else:
                alpha *= beta
        return self.alpha * p
