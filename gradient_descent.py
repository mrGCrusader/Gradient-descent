import typing as tp
from typing import Optional

import numpy as np

import learning_rate_scheduling.schedulers as lrs
import stop_criteria as sc
from gradient import find_gradient


class gradient_descent:

    def __init__(self,
            dimension: int = 2,
            function: tp.Callable[[np.array], float] = lambda arg: np.sum(np.sin(arg)),
            gradient: tp.Callable[[np.array], np.array] = lambda arg: np.cos(arg),
            test_criterion: sc.StoppingCriterion = sc.MaxIterations(),
            learning_rate_scheduling: lrs.LRScheduler = lrs.Constant()):

        """
        are you serious man? хочешь написать комментарий для init????
        разве что стоит сказать, что в изначальную функцию значения стоит передавать 
        в виде np.array. Ну и learning_rate_scheduling это правило, по которому мы делаем следующий шаг
        """
        self.dimension = dimension
        self.function = function
        self.gradient = gradient
        self.test = test_criterion
        self.lrs = learning_rate_scheduling
        self.iterations = 0
        self.function_calls = 0
        self.gradient_calls = 0
        self.points = []
    
    
    def make_min_value(self,
                       begining_point: np.array = None) -> np.array:
        """
        search min value with gradient descent.
        beginning_point: the point from which the algorithm starts working
        loging: print history in stdoutput
        return: returns the point obtained as a result of the algorithm
        """

        if begining_point is None:
            begining_point = np.random.random(self.dimension)
            # begining_point = np.array([3.0, 4.0])
            # print(f"begining_point: {begining_point}")

        if self.gradient is None:
            find_grad = find_gradient(self.function)
            self.gradient = find_grad.get_value

        step_number: int = 1
        curr_value: np.typing.NDArray = begining_point.copy()

        while not self.test.should_stop(value=self.function(curr_value),
                                        gradient=self.gradient,
                                        iteration=step_number,
                                        step=curr_value):


            cur_step = self.lrs.get_lr(iter_number=step_number,
                                       x=curr_value,
                                       p=-self.gradient(curr_value))
            lrs_logs = self.lrs.get_logs()
            self.iterations += lrs_logs[0] + 1
            self.function_calls += lrs_logs[1] + 1
            self.gradient_calls += lrs_logs[2] + 1

            curr_value -= cur_step * self.gradient(curr_value)
            step_number += 1
            self.points.append((step_number, curr_value.copy()))
        # print(step_number)
        return curr_value

    def get_logs(self) -> tuple[float, float, float]:
        return self.iterations, self.function_calls, self.gradient_calls

    def get_enum_point(self) -> np.array:
        return self.points