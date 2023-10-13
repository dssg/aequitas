from typing import Optional

import pandas as pd

from .postprocessing import PostProcessing


class Identity(PostProcessing):
    def __init__(self):
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> None:
        pass

    def transform(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> pd.Series:
        return y_hat
