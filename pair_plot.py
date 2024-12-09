import sys
import matplotlib.pyplot as plt
from preprocessing import load_data
from histogram import to_histogram
from scatter_plot import to_scatter


def main():
    """ Plots scatter plots for the grade repartition between two features,
    for each feature in the dataset, separated by house.
    """
    try:
        if len(sys.argv) != 2:
            raise IndexError('Please enter one argument.')

        df, features = load_data(sys.argv[1])
        features_len = len(features)
        _, axes = plt.subplots(
            nrows=features_len,
            ncols=features_len,
        )

        for i, ax_row in enumerate(axes):
            for j, cell in enumerate(ax_row):
                cell.set_xticks([])
                cell.set_yticks([])
                cell.set_xlabel(
                    '\n'.join(features[j].split()),
                    fontsize=7
                )
                cell.set_ylabel(
                    '\n'.join(features[i].split()),
                    fontsize=7
                )

                if i != j:
                    to_scatter(
                        df=df,
                        x_feature=features[i],
                        y_feature=features[j],
                        ax=cell
                    )
                else:
                    to_histogram(
                        df=df,
                        feature=features[i],
                        ax=cell
                    )

        for ax in axes.flat:  # Remove the labels in the middle of the plot
            ax.label_outer()

        plt.show()

    except BaseException as e:
        print(type(e).__name__, ':', e)


if __name__ == '__main__':
    main()
