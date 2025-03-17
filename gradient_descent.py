import typing as tp
from typing import Optional

import numpy as np

import learning_rate_scheduling.schedulers as lrs
import stop_criteria as sc


class gradient_descent:
    DELTA = 1e-6

    def __init__(self,
            dimension: int = 2,
            function: tp.Callable[[np.array], float] = lambda arg: np.sum(np.sin(arg)),
            gradient: tp.Callable[[np.array], np.array] = lambda arg: np.cos(arg),
            test_criterion: sc.StoppingCriterion = sc.Convergence(),
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
        self.logs: list = []
    
    def __find_gradient(self, point: np.array) -> np.array:
        default_value = self.function(point)
        anw = np.zeros_like(point)
        
        for i in range(self.dimension):
            point[i] += self.DELTA
            anw[i] = self.function(point) - default_value
            point[i] -= self.DELTA
        
        return anw / self.DELTA
    
    def make_min_value(self,
                       begining_point: np.array = None) -> np.array:
        """
        search min value with gradient descent.
        beginning_point: the point from which the algorithm starts working
        loging: print history in stdoutput
        return: returns the point obtained as a result of the algorithm
        """
        self.logs = []

        if begining_point is None:
            begining_point = np.random.random(self.dimension)
        if self.gradient is None:
            self.gradient = self.__find_gradient

        step_number: int = 1
        curr_value: np.typing.NDArray = begining_point.copy()

        while not self.test.should_stop(value=self.function(curr_value), gradient=self.gradient, iteration=step_number):
            cur_step = self.lrs.get_lr(step_number)
            print(cur_step)
            curr_value -= cur_step * self.gradient(curr_value)
            step_number += 1
            self.logs.append((step_number, curr_value.copy()))
        return curr_value


    def get_log(self) -> Optional[list]:
        if self.logs is []:
            print("No logs yet")
            return
        return self.logs







