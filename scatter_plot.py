import sys
import matplotlib.pyplot as plt
from preprocessing import load_data
from utils import get_house_color


def to_scatter(df, x_feature: str, y_feature: str, ax: plt.axes):
    """ Plots a scatter plot for the grade repartition between two features,
    separated by house.
    """
    ax.scatter(
        x=df[x_feature],
        y=df[y_feature],
        c=[
            get_house_color(house)
            for house in df['Hogwarts House']
        ],
        s=1
    )


def main():
    try:
        if len(sys.argv) != 2:
            raise IndexError('Please enter one argument.')

        df, features = load_data(sys.argv[1])
        features = df.select_dtypes('number').columns.tolist()
        for feature in features:
            for feature2 in features:
                _, ax = plt.subplots()
                to_scatter(
                    df=df,
                    x_feature=feature,
                    y_feature=feature2,
                    ax=ax
                )
                plt.xlabel(feature)
                plt.ylabel(feature2)
                plt.show()

    except BaseException as e:
        print(type(e).__name__, ':', e)


if __name__ == '__main__':
    main()
