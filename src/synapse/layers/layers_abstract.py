from abc import ABC, abstractmethod
from typing import Iterable, Protocol, Callable, Tuple
import numpy as np

class Layer(ABC):
    @abstractmethod
    def forward_prop(self, x) -> np.ndarray:
        ...
    @abstractmethod
    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        ...

class Trainable(Protocol):
    @property
    def params(self) -> Iterable[np.ndarray]: ...
    @property
    def grads(self) -> Iterable[np.ndarray]: ...