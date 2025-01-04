import numpy as np
from typing import Dict, List
import torch
from safetensors.torch import save_file, safe_open


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 42_000):
        """ Constructor for the LogisticRegression class.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights: Dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fit the model to the training data.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Inputs and outputs must be vectors of same length'
            )

        m, n = X.shape  # m = number of samples, n = number of features
        y = y.reshape(m, 1)
        classes = np.unique(y)

        for c in classes:
            weights_c = np.random.uniform(
                -np.sqrt(6 / n), np.sqrt(6 / n),
                (n, 1)
            )

            y_c = np.where(y == c, 1, 0)
            for _ in range(self.max_iter):
                z = np.dot(X, weights_c)
                predictions = self.sigmoid(z)
                predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

                # loss = np.sum(-y_c * np.log(predictions)
                #               - (1 - y_c) * np.log(1 - predictions)) / m

                # print(f'Loss: {loss}')

                d_weight = (1 / m) * np.dot(X.T, predictions - y_c)
                weights_c -= self.learning_rate * d_weight

            self.weights[c] = weights_c

    def predict(self, X: np.ndarray) -> List[Dict[str, float]]:
        """ Predict the output for the given input.
        """
        predictions = []
        for entry in X:
            predictions.append({
                c: self.sigmoid(np.dot(entry, self.weights[c]))
                for c in self.weights.keys()
            })
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Return the accuracy of the model on the given data.
        """
        predictions = self.predict(X)
        correct = 0
        for i, entry in enumerate(X):
            predicted = max(predictions[i], key=predictions[i].get)
            actual = y[i]
            if predicted == actual:
                correct += 1
        return correct / len(X) * 100

    def save(self, path: str) -> None:
        """ Save the model to the given path.
        """
        model = {
            k: torch.from_numpy(v).to(torch.float32)
            for k, v in self.weights.items()
        }
        save_file(model, path)

    def load(self, path: str) -> None:
        """ Load the model from the given path.
        """
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                self.weights = {
                    k: f.get_tensor(k).numpy()
                    for k in f.keys()
                }

        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """ Sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
