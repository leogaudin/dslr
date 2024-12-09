import pandas as pd
from stats import (
    ft_mean,
    ft_std,
    ft_min,
    ft_25,
    ft_median,
    ft_75,
    ft_max,
    ft_var,
)
from preprocessing import load_data
import sys


def main():
    """Entry point. Loads the .csv and converts it to a pandas
    dataframe to process it.
    """
    try:
        if len(sys.argv) != 2:
            raise IndexError('Please enter one argument.')

        df, features = load_data(sys.argv[1])

        describe_df = pd.DataFrame(columns=['Stat name', *features])

        operations = {
            'Count': len,
            'Mean': ft_mean,
            'Std': ft_std,
            'Var': ft_var,
            'Min': ft_min,
            '25%': ft_25,
            '50%': ft_median,
            '75%': ft_75,
            'Max': ft_max,
        }

        # Perform every operation on every column
        stats = [
            [
                operation,
                *[  # "*" is the spread operator, like "...array" in JavaScript
                    operations[operation](df[column])
                    for column in features
                ]
            ] for operation in operations.keys()
        ]

        # Append every stat to the describe dataframe
        for stat in stats:
            describe_df.loc[len(describe_df)] = stat

        print(describe_df)
    except BaseException as e:
        print(type(e).__name__, ':', e)


if __name__ == "__main__":
    main()
