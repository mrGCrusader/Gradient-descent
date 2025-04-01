import typing as tp

import numpy as np

import stop_criteria as sc
from gradient import find_gradient
from learning_rate_scheduling import LRScheduler


class ArmijoRule(LRScheduler):
    def __init__(self, dimension: int = 2,
                 function: tp.Callable[[np.array], float] = lambda arg: np.sum(np.sin(arg)),
                 gradient: tp.Callable[[np.array], np.array] = None,
                 test_criterion: sc.StoppingCriterion = sc.Combine(stop1=sc.MaxIterations(1000), stop2=sc.Convergence()),
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
                       beta=0.5, c1=0.1,  **kwargs):

        alpha = self.alpha
        # print(f"alpha Armijo rule: {alpha}")
        fun_value_x = self.function(x)
        self.function_calls += 1

        if x is None:
            x = np.random.random(self.dimension)
            # print(f"x Armijo: {x}")


        if self.gradient is None:
            find_grad = find_gradient(self.function)
            self.gradient = find_grad.get_value
        grad_x = self.gradient(x)
        self.gradient_calls += 1


        # print(f"grad_x: {grad_x}")
        if p is None:
            p = -grad_x
        # print(f"p Armijo: {p}")


        step_number: int = 1
        curr_value: np.typing.NDArray = x.copy()
        grad_dot_p = grad_x.dot(p)

        if grad_dot_p >= 0:
            p = -grad_x
            grad_dot_p = grad_x.dot(p)

        while not self.test.should_stop(step=curr_value,
                                        value=self.function(curr_value),
                                        gradient=grad_x,
                                        iteration=step_number):
            self.iterations += 1
            curr_value = x + alpha * p
            fx_new = self.function(curr_value)
            self.function_calls += 1

            armijo_condition = fx_new <= fun_value_x + c1 * alpha * grad_dot_p

            if armijo_condition:
                return alpha
            else:
                alpha *= beta
        return alpha