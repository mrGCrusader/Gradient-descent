import math

"""
Идея: держим шаг постоянным до тех пор, пока не определим, что мы застряли в одном месте, и не
уменьшим шаг. -- надо сделать
"""


class LRScheduler:
    """
    Abstract base class for learning rate schedulers.
    """

    def __init__(self, initial_lr: float = 0.1):
        self.initial_lr = initial_lr

    def get_lr(self, iter_number=None):
        raise NotImplementedError("Subclasses must implement this method")


class Constant(LRScheduler):
    """
    just a constant learning rate
    """

    def get_lr(self, iter_number: int = 0):
        return self.initial_lr


class TimeBasedDecay(LRScheduler):
    """
    h(k + 1) = h(k) / (1 + l * k)
    """

    def __init__(self, initial_lr: float = 0.1, hyper_lambda: int = 1):
        super().__init__(initial_lr)
        self.last = initial_lr
        self.hyper_lambda = hyper_lambda

    def get_lr(self, iter_number: int = 0):
        self.last = self.last / (1 + self.hyper_lambda * iter_number)
        return self.last


class StepDecay(LRScheduler):
    """
    h(k) = h_0 * d ** floor((1 + k) // step_size)
    """

    def __init__(self, initial_lr: float = 0.1, hyper_base: int = 2, hyper_lambda: int = 1):
        super().__init__(initial_lr)
        self.hyper_base = hyper_base
        self.step_size = hyper_lambda

    def get_lr(self, iter_number: int = 0):
        return self.initial_lr * (self.hyper_base ** math.floor((1 + iter_number) // self.step_size))


class ExponentialDecay(LRScheduler):
    """
    h(k) = h_0 * exp(-l * k)
    """

    def __init__(self, initial_lr: float = 0.1, hyper_lambda: int = 1):
        super().__init__(initial_lr)
        self.hyper_lambda = hyper_lambda

    def get_lr(self, iter_number: int = 0):
        return self.initial_lr * math.exp(-self.hyper_lambda * iter_number)


class PolynomialDecay(LRScheduler):
    """
    h(k) = h_0 * (beta * k + 1) ** (-alpha)
    """

    def __init__(self, initial_lr: float = 0.1, hyper_alpha: float = 0.5, hyper_beta: float = 1):
        super().__init__(initial_lr)
        self.hyper_alpha = hyper_alpha
        self.hyper_beta = hyper_beta

    def get_lr(self, iter_number: int = 0):
        return self.initial_lr * (self.hyper_beta * iter_number + 1) ** (-self.hyper_alpha)

# class CosineAnnealingLR:
#     def __init__(self, initial_lr, T_max):
#         self.initial_lr = initial_lr
#         self.T_max = T_max
#
#     def get_lr(self, iter_num):
#         return self.initial_lr * (1 + math.cos(math.pi * iter_num / self.T_max)) / 2
