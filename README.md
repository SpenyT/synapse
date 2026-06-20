# Synapse

A neural network library built from scratch using nothing but NumPy. No PyTorch. No TensorFlow. No autograd magic — just math, matrices, and a genuine desire to understand what happens inside a neural network.

The goal of this project was to learn AI fundamentals by implementing them: forward propagation, backpropagation, weight initialization, activation functions, loss functions, and gradient descent, all from first principles.

---

## AI fundamentals covered

Building Synapse from scratch meant implementing these core ideas by hand:

| Concept | Where |
|---|---|
| Layer abstraction / protocols | `layers_abstract.py` |
| Weight initialization (Xavier, He, LeCun) | `initializations.py` |
| Activation functions + their derivatives | `activations.py` |
| Forward propagation | `fully_connected_layer.py` |
| Backpropagation (chain rule) | `fully_connected_layer.py` |
| Loss functions (MSE, Cross-Entropy) | `loss.py` |
| Softmax for multi-class output | `loss.py` |
| Training loop with gradient descent | `neural_network.py` |
| Prediction + accuracy evaluation | `neural_network.py` |

---

## Examples

### Digit classifier — `examples/digits_cnn.py`

Trains a fully-connected network on scikit-learn's built-in MNIST-style digits dataset (8×8 grayscale images, classes 0–9).

**Architecture:** `64 → 64 (ReLU) → 10`

### Alphabet classifier — `examples/alphabet_cnn.py`

Trains a deeper network on two Kaggle handwritten letter datasets (28×28 images, classes A–Z). The second dataset is used as a holdout to test how well the model generalizes.

**Architecture:** `784 → 142 (ReLU) → 142 (ReLU) → 26`

---

## Requirements

- Python 3.10+

Dependencies (managed via `pyproject.toml`):

| Package | Purpose |
|---|---|
| `numpy` | All matrix math and numerical operations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Dataset utilities and train/test splits |
| `kagglehub` | Downloading datasets from Kaggle |
| `matplotlib` | Visualizing predictions |
| `tqdm` | Progress bars during training |

---

## Getting started

**1. Clone the repo**

```bash
git clone https://github.com/SpenyT/synapse.git
cd synapse
```

**2. Install in editable mode**

This installs the `synapse` package locally so the examples can import it from anywhere.

```bash
python -m pip install -e .
```

**3. Run an example**

```bash
# Digit classifier (no credentials needed)
python examples/digits_cnn.py

# Alphabet classifier (requires Kaggle API key)
python examples/alphabet_cnn.py
```

---


## License

MIT — Spencer Toupin
