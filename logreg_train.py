import sys
from preprocessing import load_data, train_test_split
import numpy as np
from model import LogisticRegression


def main():
    try:
        if len(sys.argv) != 2:
            raise IndexError('Please enter one argument.')

        df, features = load_data(sys.argv[1])

        X = np.asarray(df[features])
        Y = np.asarray(df['Hogwarts House'])

        (
            X_train,
            X_test,
            Y_train,
            Y_test
        ) = train_test_split(X, Y, test_size=0)

        model = LogisticRegression()
        model.fit(X_train, Y_train)
        # print(model.score(X_train, Y_train))
        model.save('weights')

    except BaseException as e:
        print(type(e).__name__, ':', e)


if __name__ == '__main__':
    main()
