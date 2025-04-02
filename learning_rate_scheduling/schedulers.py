import math
import typing as tp


class LRScheduler:
    """
    Abstract base class for learning rate schedulers.
    """

    def __init__(self, initial_lr: float = 0.1, dimension: int = 2):
        self.initial_lr = initial_lr
        self.dimension = dimension
        self.iterations = 0
        self.function_calls = 0
        self.gradient_calls = 0

    def get_lr(self, iter_number: tp.Optional[int] = None, **kwargs) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    def get_logs(self) -> tuple[int, int, int]:
        """Returns (iterations_count, function_calls, gradient_calls)"""
        return self.iterations, self.function_calls, self.gradient_calls

    def _update_counters(self, iter_number: int):
        self.iterations = iter_number if iter_number is not None else self.iterations + 1


class Constant(LRScheduler):
    """Constant learning rate"""

    def get_lr(self, iter_number: int = None, **kwargs) -> float:
        self._update_counters(iter_number)
        return self.initial_lr


class TimeBasedDecay(LRScheduler):
    """h(k) = h0 / (1 + λ*k)"""

    def __init__(self, initial_lr: float = 0.1, decay_rate: float = 0.1):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def get_lr(self, iter_number: int = None, **kwargs) -> float:
        self._update_counters(iter_number)
        k = self.iterations
        return self.initial_lr / (1 + self.decay_rate * k)


class StepDecay(LRScheduler):
    """h(k) = h0 * β^floor(k/step)"""

    def __init__(self, initial_lr: float = 0.1, step_size: int = 10, decay_factor: float = 0.5):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.decay_factor = decay_factor

    def get_lr(self, iter_number: int = None, **kwargs) -> float:
        self._update_counters(iter_number)
        k = self.iterations
        return self.initial_lr * (self.decay_factor ** math.floor(k / self.step_size))


class ExponentialDecay(LRScheduler):
    """h(k) = h0 * exp(-λ*k)"""

    def __init__(self, initial_lr: float = 0.1, decay_rate: float = 0.1):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def get_lr(self, iter_number: int = None, **kwargs) -> float:
        self._update_counters(iter_number)
        k = self.iterations
        return self.initial_lr * math.exp(-self.decay_rate * k)


class PolynomialDecay(LRScheduler):
    """h(k) = h0 * (β*k + 1)^(-α)"""

    def __init__(self, initial_lr: float = 0.1, alpha: float = 0.5, beta: float = 1.0):
        super().__init__(initial_lr)
        self.alpha = alpha
        self.beta = beta

    def get_lr(self, iter_number: int = None, **kwargs) -> float:
        self._update_counters(iter_number)
        k = self.iterations
        return self.initial_lr * ((self.beta * k + 1) ** (-self.alpha))


class CosineAnnealingLR(LRScheduler):
    """h(k) = h0*(1 + cos(π*k/T))/2"""

    def __init__(self, initial_lr: float = 0.1, cycle_length: int = 10):
        super().__init__(initial_lr)
        self.cycle_length = cycle_length

    def get_lr(self, iter_number: int = None, **kwargs) -> float:
        self._update_counters(iter_number)
        k = self.iterations
        return self.initial_lr * (1 + math.cos(math.pi * k / self.cycle_length)) / 2
