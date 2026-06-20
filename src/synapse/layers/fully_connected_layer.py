import numpy as np
from typing import Optional, Callable, Tuple
from .utils.initializations import resolve_weight_initializer, resolve_bias_initializer
from .utils.activations import resolve_activation
from .layers_abstract import Layer

class FCL(Layer):
    def __init__(self, n_input: int, n_output : int, activation: str = None, weight_initializer: str = "he", bias_initializer: str = "zeroes"):
        self._activation_fn: Callable[[np.ndarray], np.ndarray] = None 
        self._activation_deriv: Callable[[np.ndarray], np.ndarray] = None
        self._activation_fn, self._activation_deriv = resolve_activation(activation)
        self.W = np.random.randn(n_output, n_input) * resolve_weight_initializer(weight_initializer, n_input, n_output)
        self.b = resolve_bias_initializer(bias_initializer, n_output)
        self._dW = np.zeros_like(self.W)
        self._db = np.zeros_like(self.b)
        self._input: Optional[np.ndarray] = None

    @property
    def params(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.W, self.b)
    @property
    def grads(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._dW, self._db)
    
    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        self._input = X
        self._Z = X @ self.W.T + self.b.T
        return self._activation_fn(self._Z)

    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        local = self._activation_deriv(self._Z)
        if local.shape != gradient_output.shape:
            print("ERROR: derivation shape: ", local.shape, "!= gradient output shape: ", gradient_output.shape)
        dZ = gradient_output * local

        self._dW = dZ.T @ self._input
        self._db = dZ.sum(axis=0, keepdims=True).T

        grad_input = dZ @ self.W

        self.W -= learning_rate * self._dW
        self.b -= learning_rate * self._db

        return grad_input