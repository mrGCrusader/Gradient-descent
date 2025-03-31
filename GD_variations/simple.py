import typing as tp
from typing import Optional, override

import numpy as np

import learning_rate_scheduling.schedulers as lrs
import stop_criteria as sc
from gradient import find_gradient


class SimpleSearch:

    def __init__(self,
                 dimension: int = 2,
                 function: tp.Callable[[np.array], float] = lambda arg: np.sum(np.sin(arg)),
                 gradient: tp.Callable[[np.array], np.array] = None,
                 test_criterion: sc.StoppingCriterion = sc.Convergence(),
                 learning_rate_scheduling: lrs.LRScheduler = lrs.Constant()):
        self.dimension = dimension
        self.function = function
        self.gradient = gradient
        self.test = test_criterion
        self.lrs = learning_rate_scheduling
    def make_min_value(self,
                       begining_point: np.array = None) -> np.array:
        """
        search min value with gradient descent.
        beginning_point: the point from which the algorithm starts working
        return: returns the point obtained as a result of the algorithm
        """

        if begining_point is None:
            begining_point = np.random.random(self.dimension)
        if self.gradient is None:
            find_grad = find_gradient(self.function)
            self.gradient = find_grad.get_value

        step_number: int = 1
        curr_value: np.typing.NDArray = begining_point.copy()

        while not self.test.should_stop(step=curr_value,
                                        value=self.function(curr_value),
                                        gradient=self.gradient,
                                        iteration=step_number):
            cur_step = self.lrs.get_lr(step_number)
            curr_value -= cur_step * self.gradient(curr_value)
            step_number += 1
        return curr_value
