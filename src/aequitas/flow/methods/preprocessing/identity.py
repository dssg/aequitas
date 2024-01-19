from typing import Optional

import pandas as pd

from .preprocessing import PreProcessing


class Identity(PreProcessing):
    def __init__(self):
        self.used_in_inference = False
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        pass

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        return X, y, s
