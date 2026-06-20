import numpy as np
from matplotlib import pyplot as plt
import random

from synapse.models.neural_network import NeuralNetwork

# shows one example
def show_example(net: NeuralNetwork, X: np.ndarray, y: np.ndarray, idx: int, img_size: tuple[int, int] =(28, 28), as_letter: bool =False):
    x1 = X[idx].reshape(1, -1)
    pred = int(net.predict(x1)[0])
    label = int(y[idx])
    
    def to_char(v): return chr(v + 65)
    pred_str  = to_char(pred)  if as_letter else str(pred)
    label_str = to_char(label) if as_letter else str(label)

    plt.imshow(X[idx].reshape(*img_size), cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(f"Pred: {pred_str}  |  Label: {label_str}")
    plt.show()

# shows multiple examples at a time
def show_grid(X: np.ndarray, y: np.ndarray, preds: np.ndarray, rows: int, cols: int, img_size: tuple[int, int], cmap: str ="gray"):
    n = min(len(X), rows * cols)
    idx = np.random.choice(len(X), n, replace=False)

    imgs = X[idx]
    if imgs.ndim == 2:
        imgs = imgs.reshape(n, *img_size)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, idx):
        ax.imshow(X[i].reshape(*img_size), cmap=cmap)
        ax.axis("off")
        title = []
        if y is not None:
            title.append(f"label:{y[i]}")
        if preds is not None:
            title.append(f"pred:{preds[i]}")
        if title:
            ax.set_title(" | ".join(title), fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_random_predictions_once(net: NeuralNetwork, X: np.ndarray, y: np.ndarray, n: int =5, img_size: tuple[int, int] =(28, 28), as_letter: bool =False, seed=None):
    rng = random.Random(seed)
    indices = rng.sample(range(len(y)), k=min(n, len(y)))

    for idx in indices:
        show_example(net, X, y, idx, img_size=img_size, as_letter=as_letter)

def show_random_predictions(
    net: NeuralNetwork, 
    X: np.ndarray, 
    y: np.ndarray, 
    rows: int, 
    cols: int, 
    n: int =None,
    img_size: tuple[int, int] =(28, 28), 
    as_letter: bool=False, 
    seed=None,
):
    rng = random.Random(seed)
    n = min(len(y), rows * cols) if n is None else min(n, len(y), rows * cols)
    idx = np.array(rng.sample(range(len(y)), k=n))

    Xs = X[idx]
    ys = y[idx]
    preds = net.predict(Xs)

    if as_letter:
        to_char = np.vectorize(lambda v: chr(int(v) + 65))
        ys_disp = to_char(ys.astype(int))
        preds_disp = to_char(preds.astype(int))
    else:
        ys_disp = ys
        preds_disp = preds

    show_grid(Xs, ys_disp, preds_disp, rows, cols, img_size, cmap="gray")