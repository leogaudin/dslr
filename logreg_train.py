import sys
from preprocessing import load_data, train_test_split
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypothesis(
    weights: np.ndarray,
    inputs: np.ndarray[np.ndarray]
):
    return sigmoid(np.dot(inputs, weights))


def train(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    max_iter: int,
):
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            'Inputs and outputs must be vectors of same length'
        )

    m = x.shape[0]
    n = x.shape[1]

    y = y.reshape(m, 1)

    classes = np.unique(y)

    # Shapes recap
    #
    # x             m rows, n columns
    # y             m rows, 1 column
    # weights       n rows, 1 column
    # z             m rows, 1 column
    # predictions   m rows, 1 column
    # d_weight      n rows, 1 column

    w = {}

    for c in classes:
        weights = np.zeros((n, 1))
        # cost_list = []
        y_c = np.where(y == c, 1, 0)
        for _ in range(max_iter):
            # Because np.dot(weights.T, x) does not work
            # and I still don't understand why
            z = np.dot(x, weights)
            predictions = sigmoid(z)
            predictions = np.clip(predictions, 1e-16, 1 - 1e-16)

            # cost = -(1 / m) * np.sum(
            #     np.multiply(
            #         y_c,
            #         np.log(predictions)
            #     )
            #     + np.multiply(
            #         1 - y_c,
            #         np.log(1 - predictions)
            #     )
            # )

            # cost_list.append(cost)

            d_weight = (1 / m) * np.dot(x.T, predictions - y_c)

            weights -= learning_rate * d_weight

        w[c] = weights

    return w


def main():
    """
    """
# try:
    if len(sys.argv) != 2:
        raise IndexError('Please enter one argument.')

    df, features = load_data(sys.argv[1])
    X = np.asarray(df[features])
    Y = np.asarray(df['Hogwarts House'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    w = train(
        x=X_train,
        y=Y_train,
        learning_rate=0.001,
        max_iter=50000
    )

    correct = 0
    for i, entry in enumerate(X_test):
        predictions = {
            c: hypothesis(w[c], entry)
            for c in w.keys()
        }
        predicted = max(predictions, key=predictions.get)
        actual = Y_test[i]
        if predicted == actual:
            correct += 1

    print(f'Accuracy: {correct / len(X_test) * 100:.2f}%')

# except BaseException as e:
#     print(type(e).__name__, ':', e)


if __name__ == '__main__':
    main()
