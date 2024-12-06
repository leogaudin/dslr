import sys
import matplotlib.pyplot as plt
from preprocessing import load_data
from utils import houses, get_house_color


def to_histogram(df, feature: str, ax: plt.axes):
    """ Plots a histogram for the grade repartition of a feature,
    separated by house.
    """
    for house in houses:
        ax.hist(
            df.loc[df['Hogwarts House'] == house, feature],
            alpha=.5,
            label=house,
            color=get_house_color(house)
        )


def main():
    try:
        if len(sys.argv) != 2:
            raise IndexError('Please enter one argument.')

        df = load_data(sys.argv[1])
        feature_columns = df.select_dtypes('number').columns.tolist()
        for feature in feature_columns:
            _, ax = plt.subplots()
            to_histogram(df, feature, ax)
            plt.title(feature)
            plt.xlabel('Grade')
            plt.ylabel('Number of students')
            plt.legend(title='House')
            plt.show()

    except BaseException as e:
        print(type(e).__name__, ':', e)


if __name__ == '__main__':
    main()
