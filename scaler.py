import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def save(self, filename):
        np.savez(filename, mean=self.mean, std=self.std)

    def load(self, filename):
        data = np.load(filename)
        if 'mean' not in data or 'std' not in data:
            raise ValueError('No mean or std in the file.')
        self.mean = data['mean']
        self.std = data['std']

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean
