import numpy as np
from typing import Callable, Tuple

def resolve_loss(
    loss_name: str
) -> Tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray], np.ndarray]
]:
    match loss_name:
        case "mse":
            return mse, mse_prime
        case "cross_entropy":
            return cross_entropy, cross_entropy_prime
        case _:
            raise ValueError(f"Loss function '{loss_name}' not recognized.")


def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z))
    A = Z_exp / Z_exp.sum(axis=0, keepdims=True)
    return A

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))
            
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true: np.ndarray, logits: np.ndarray):
    B, _ = logits.shape
    p = _softmax_rowwise(logits)
    y_idx = _ensure_sparse_targets(y_true)
    eps = 1e-12
    ll = -np.log(p[np.arange(B), y_idx] + eps)
    return float(np.mean(ll))
            
def cross_entropy_prime(y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
    B, _ = logits.shape
    p = _softmax_rowwise(logits)
    y_idx = _ensure_sparse_targets(y_true)

    grad = p.copy()
    grad[np.arange(B), y_idx] -= 1.0
    grad /= B
    return grad



# ----- helpers -----
def _softmax_rowwise(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

# honestly not fully sure what's going on here but will be reading more about it
def _ensure_sparse_targets(y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    if y_true.ndim == 2 and y_true.shape[1] == 1 and np.issubdtype(y_true.dtype, np.integer):
        return y_true.ravel()
    if y_true.ndim == 1 and np.issubdtype(y_true.dtype, np.integer):
        return y_true
    if y_true.ndim == 2:
        return np.argmax(y_true, axis=1)
    raise ValueError("y_true has wrong format for cross-entropy.")