import numpy as np
from typing import Callable, Tuple

def resolve_activation(
    activation_name: str
) -> Tuple[
    Callable[[np.ndarray], np.ndarray], 
    Callable[[np.ndarray], np.ndarray]
]:
    match activation_name:
        case "relu":
            return relu, relu_prime
        case "sigmoid":
            return sigmoid, sigmoid_prime
        case "tanh":
            return tanh, tanh_prime
        case None:
            return lambda x: x, lambda x: np.ones_like(x)
        case _:
            raise ValueError(f"Activation function '{activation_name}' not recognized.")
        

def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)

def relu_prime(Z: np.ndarray) -> np.ndarray:
    return (Z > 0).astype(float)

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z: np.ndarray) -> np.ndarray:
    s = sigmoid(Z)
    return s * (1 - s)

def tanh(Z: np.ndarray) -> np.ndarray:
    return np.tanh(Z)

def tanh_prime(Z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(Z) ** 2