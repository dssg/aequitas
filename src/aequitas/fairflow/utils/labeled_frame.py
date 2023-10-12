import warnings

import pandas as pd


class LabeledFrame(pd.DataFrame):
    """Wrapper class that adds a few convenience properties to a DataFrame.

    Parameters
    ----------
    pd : DataFrame
        The DataFrame to wrap.
    y_col : str, optional
        The name of the target column. Defaults to None.
    s_col : str, optional
        The name of the sensitive column. Defaults to None.
    ignore_cols : list[str], optional
        A list of column names to ignore. Defaults to [].

    """

    def __init__(self, *args, y_col=None, s_col=None, ignore_cols=[], **kwargs):
        super().__init__(*args, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            self.y_col = y_col
            self.s_col = s_col
            self.ignore_cols = ignore_cols

    @property
    def X(self):
        return self.drop(columns=[self.y_col, self.s_col] + self.ignore_cols)

    @property
    def y(self):
        return self[self.y_col]

    @property
    def s(self):
        return self[self.s_col]
