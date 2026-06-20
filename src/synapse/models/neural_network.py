import numpy as np
from tqdm import tqdm
from synapse.layers import Layer
from synapse.models.utils.loss import resolve_loss


class NeuralNetwork:
    def __init__(self, model : np.ndarray[Layer] = [], loss: str = "mse"):
        self.layers = model
        self.loss, self.loss_prime = resolve_loss(loss)

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(
                f"Cannot add object of type {type(layer)}. "
                f"Only objects that inherit from 'Layer' are supported."
            )
        self.layers.append(layer)

    def forward_prop(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_prop(input_data)
        return input_data
    
    def backward_prop(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_prop(loss_grad, learning_rate)

    def train(self, x_train, y_train, epochs: int, learning_rate: float, print_output: bool = True):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        n = len(x_train)

        if print_output:
            print("Starting training...")

        for epoch in range(epochs):
            err_sum = 0.0
            correct = 0

            it = zip(x_train, y_train)
            if print_output:
                it = tqdm(it, total=n, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for x, y in it:
                x = x.reshape(1, -1)
                logits = self.forward_prop(x)

                loss = self.loss(np.array([y]), logits)
                err_sum += float(loss)
                grad = self.loss_prime(np.array([y]), logits)
                self.backward_prop(grad, learning_rate)

                pred = int(np.argmax(logits, axis=1)[0])
                if pred == int(y):
                    correct += 1

                if print_output:
                    seen = correct + (0)
                    processed = it.n if hasattr(it, "n") else 1
                    avg_loss = err_sum / max(processed, 1)
                    acc = correct / max(processed, 1)
                    it.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

            err = err_sum / n
            train_acc = correct / n
            if print_output:
                tqdm.write(f"Epoch {epoch+1}/{epochs} - loss: {err:.4f} - acc: {train_acc:.4f}")
                pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward_prop(np.asarray(X))
        return np.argmax(logits, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float((preds == y).mean())
