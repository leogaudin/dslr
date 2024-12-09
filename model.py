import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.001, max_iter=50000):
        """ Constructor for the LogisticRegression class.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        """ Fit the model to the training data.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Inputs and outputs must be vectors of same length'
            )

        m = X.shape[0]
        n = X.shape[1]

        y = y.reshape(m, 1)

        classes = np.unique(y)

        self.weights = {}

        for c in classes:
            weights_c = np.zeros((n, 1))
            y_c = np.where(y == c, 1, 0)
            for _ in range(self.max_iter):
                z = np.dot(X, weights_c)
                predictions = self.sigmoid(z)
                predictions = np.clip(predictions, 1e-13, 1 - 1e-13)

                d_weight = (1 / m) * np.dot(X.T, predictions - y_c)

                weights_c -= self.learning_rate * d_weight

            self.weights[c] = weights_c

    def predict(self, X):
        """ Predict the output for the given input.
        """
        predictions = []
        for entry in X:
            predictions.append({
                c: self.sigmoid(np.dot(entry, self.weights[c]))
                for c in self.weights.keys()
            })

        return predictions

    def score(self, X, y):
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

    def save(self, path):
        """ Save the model to the given path.
        """
        np.save(path, self.weights)

    def load(self, path):
        """ Load the model from the given path.
        """
        self.weights = np.load(path, allow_pickle=True).item()

    @staticmethod
    def sigmoid(z):
        """ Sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
