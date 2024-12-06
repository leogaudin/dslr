import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """ Loads the .csv at the given path and returns a pandas dataframe.
    """
    df = pd.read_csv(path, index_col=False)
    del df['Index']  # Delete Index column
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    pd.options.display.float_format = '{:.1f}'.format  # 2 decimals only

    return df
