import pandas as pd
import numpy as np


def load_data(
    path: str,
    dropna: bool = True
) -> pd.DataFrame:
    """ Loads the .csv at the given path and returns a pandas dataframe.
    """
    df = pd.read_csv(path, index_col=False)

    excluded = [
        'Index',
        'Hogwarts House',
        'Arithmancy',
        'Care of Magical Creatures',
        'Potions',
        'Flying',
    ]

    features = [
        f
        for f in df.select_dtypes('number').columns.tolist()
        if f not in excluded
    ]

    if dropna:
        df = df.dropna(subset=features)

    pd.options.display.float_format = '{:.1f}'.format  # 2 decimals only

    return df, features


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float
):
    if len(x) != len(y):
        raise ValueError('x and y have to be of same size.')

    if not (0 <= test_size <= 1):
        raise ValueError('test_size must be between 0 and 1.')

    n = len(x)
    n_test = int(n * test_size)
    n_train = n - n_test

    indices = np.arange(n)
    np.random.shuffle(indices)

    x_shuffle = x[indices]
    y_shuffle = y[indices]

    X_train = x_shuffle[:n_train]
    X_test = x_shuffle[n_train:]
    y_train = y_shuffle[:n_train]
    y_test = y_shuffle[n_train:]

    return X_train, X_test, y_train, y_test
