import numpy as np


class StoppingCriterion:
    """
    Abstract base class for stopping criteria.
    """

    def __init__(self):
        self.history = []

    def should_stop(self, **kwargs) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        self.history = []


class MaxIterations(StoppingCriterion):
    """
    Stops after n iterations.
    """

    def __init__(self, max_iter: int):
        super().__init__()
        self.max_iter = max_iter

    def should_stop(self, iteration: int, **kwargs) -> bool:
        return iteration >= self.max_iter


class Convergence(StoppingCriterion):
    """
    Stopping at value convergence
    (the change in the value of the value is less than the threshold).
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def should_stop(self, value: float, value_to_compare: float = None, **kwargs) -> bool:
        if value_to_compare is None:
            value_to_compare = self.eps
        self.history.remove(self.history[0])
        self.history.append(value)
        if len(self.history) < 2:
            return False
        if abs(self.history[-1] - self.history[-2]) < value_to_compare:
            return True
        return False


class StepConvergence(Convergence):
    """
    Stopping at step norm convergence
    (the change in the value of the function or step norm is less than the threshold).
    """

    def should_stop(self, step: np.typing.NDArray, **kwargs) -> bool:
        value = float(np.linalg.norm(step))
        return Convergence.should_stop(self, value, self.eps, **kwargs)


class ComparativeConvergence(Convergence):
    """
    Stopping at value comparative convergence
    (the change in the value of the value is less than the eps * |value|).
    """

    def should_stop(self, value: float, **kwargs) -> bool:
        value_to_compare = self.eps * (abs(value) + 1)
        return Convergence.should_stop(self, value, value_to_compare, **kwargs)


class StepComparativeConvergence(ComparativeConvergence):
    """
    Stopping at step norm convergence
    (the change in the value of the step norm is less than the eps * |value|).
    """

    def should_stop(self, step: np.typing.NDArray, **kwargs) -> bool:
        value = float(np.linalg.norm(step))
        return ComparativeConvergence.should_stop(self, value, **kwargs)


class GradientNorm(StoppingCriterion):
    """
    Stops if gradient norm is less than the threshold.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def should_stop(self, gradient: np.typing.NDArray, **kwargs) -> bool:
        gradient_norm = np.linalg.norm(gradient)
        return gradient_norm < self.eps


class GradientNormComparative(StoppingCriterion):
    """
    Stops if gradient norm is less than the eps * |gradient(x_0)|.
    """

    def __init__(self, eps: float = 1e-5, begin_gradient: np.typing.NDArray = 1):
        super().__init__()
        self.eps = eps
        self.begin_gradient = begin_gradient

    def should_stop(self, gradient: np.typing.NDArray, **kwargs) -> bool:
        gradient_norm = np.linalg.norm(gradient)
        return gradient_norm < self.eps * np.linalg.norm(self.begin_gradient)
