import sys
from preprocessing import load_data
import numpy as np
import pandas as pd
from model import LogisticRegression


def main():
    try:
        if len(sys.argv) != 2:
            raise IndexError('Please enter one argument.')

        df, features = load_data(sys.argv[1], dropna=False)
        X_test = np.asarray(df[features])

        model = LogisticRegression()
        model.load('weights.npy')

        results = model.predict(X_test)

        output_df = pd.DataFrame(
            data=(
                (i, max(results[i], key=results[i].get))
                for i in range(len(results))
            ),
            columns=['Index', 'Hogwarts House']
        )

        output_df.to_csv('houses.csv', index=False)

    except BaseException as e:
        print(type(e).__name__, ':', e)


if __name__ == '__main__':
    main()
