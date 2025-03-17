import numpy as np
import typing as tp

class gradient_descent:
    def __init__(self, 
            dimension: int = 2,
            function: tp.Callable[[np.array], float] = lambda arg: np.sum(np.sin(arg)), \
            gradient: tp.Callable[[np.array], np.array] = lambda arg: np.cos(arg), \
            test_criterion: tp.Callable[..., bool] = lambda count: count < 100, \
            learning_rate_sceduling: tp.Callable[[int], float] | float = 0.1):
        """
        are you serious man? хочешь написать комментарий для init????
        разве что стоит сказать, что в изначальную функцию значения стоит передавать 
        в виде np.array. Ну и learning_rate_scheduling это правило, по которому мы делаем следующий шаг
        """
        self.dimension = dimension
        self.function = function
        self.gradient = gradient
        self.test = test_criterion
        self.lrs = learning_rate_sceduling
        self.logs: list = []

    def make_min_value(self,
                       begining_point: np.array = None) -> np.array:
        """
        search min value with gradient descent.
        beginning_point: the point from which the algorithm starts working
        loging: print history in stdoutput
        return: returns the point obtained as a result of the algorithm
        """
        self.logs = []
        if (begining_point is None):
            begining_point = np.random.random(self.dimension)
        step_number: int = 1
        curr_value: np.array[float] = begining_point.copy()
        
        while (self.test(step_number)):
            curr_value -= (lrs:=self.lrs(step_number)) * self.gradient(curr_value)
            print(curr_value)
            step_number += 1
            self.logs.append((step_number, curr_value.copy()))
        return curr_value
    

    def get_log(self) -> list:
        if self.logs is []:
            print("No logs yet")
            return
        return self.logs
    

        
        
        
    
    
