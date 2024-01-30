from typing import Any, Optional

import pandas as pd

from ...utils import create_logger
from .preprocessing import PreProcessing

STRATEGIES = ["undersample", "oversample"]


class PrevalenceSampling(PreProcessing):
    def __init__(
        self,
        s_ref: Optional[Any] = "global",
        alpha: float = 1,
        strategy: str = "undersample",
        seed: int = 42,
    ):
        """Generates training sample with balanced prevalence for the groups in dataset.

        Parameters
        ----------
        s_ref : Any
            Reference group value. The global prevalence value will be obtained from
            this group. If None, the selected value will be the most frequent group.
            If "global", the reference prevalence will be calculated for the dataset.
        alpha : float, optional
            Parameter that controls the new prevalence values. A value of 0 keeps
            original prevalence, and a value of 1 equalizes the prevalence.
            Defaults to 1.
        strategy : string, optional
            Strategy of sampling. If "undersample", removes samples until reaching
            desired value. If "oversample", creates instead a bootstrap sample.
            Defaults to "undersample".
        """
        self.logger = create_logger("methods.preprocessing.PrevalenceSampling")
        self.logger.info("Instantiating a PrevalenceSampling preprocessing method.")
        # Validate sampling strategy:
        if strategy not in STRATEGIES:
            raise ValueError(f"Invalid strategy value. Try one of {STRATEGIES}.")
        self.strategy = strategy
        self.s_ref = s_ref
        self.alpha = alpha
        self.or_prevalence: dict[Any, float] = None  # type: ignore
        self.ref_prevalence: float = None  # type: ignore
        self.used_in_inference = False
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]) -> None:
        """Fits the sampler to the data.

        Updates observed prevalence in the reference group and calculates sampling rates
        for every group in the dataset.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Label vector.
        s : pandas.Series
            Protected attribute vector.
        """
        super().fit(X, y, s)
        self.logger.info("Fitting sampling method.")

        if s is None:
            raise ValueError("Sensitive Attribute `s` not passed.")

        if self.s_ref is None:  # Get more frequent group.
            self.s_ref = s.mode().values[0]

        self.or_prevalence = {}
        for group in s.unique():
            self.or_prevalence[group] = y[s == group].mean()  # Assuming 0/1 label.
        if self.s_ref == "global":
            self.ref_prevalence = y.mean()
        else:
            self.ref_prevalence = self.or_prevalence[self.s_ref]

        self.logger.debug(f"Original prevalence by group: {self.or_prevalence}")

        self.sampling_rate = {}

        for group in s.unique():
            if group == self.s_ref:
                continue  # No need to calculate sampling rates for reference group.

            positives = y[(s == group) & (y == 1)].shape[0]
            negatives = y[(s == group) & (y == 0)].shape[0]

            self.logger.debug(f"Calculating sampling size for group {group}.")

            self.sampling_rate[group] = self.calculate_sample_sizes(
                positives, negatives, self.ref_prevalence, self.strategy, self.alpha
            )
            self.logger.debug(
                f"Sampling for group {group}: {self.sampling_rate[group]}"
            )
        self.logger.info("Sampling method fitted.")

    def transform(
        self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Samples the input DataFrame and Series to

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Label vector.
        s : pd.Series, optional
            Protected attribute vector.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.Series]
            The transformed input, X, y, and s.
        """
        super().transform(X, y, s)
        
        self.logger.info("Transforming data.")
        if s is None:
            raise ValueError("Sensitive Attribute `s` not passed.")

        final_X = X.copy()
        final_y = y.copy()
        final_s = s.copy()
        for group, (label, size) in self.sampling_rate.items():
            self.logger.debug(
                f"Original group {group} size: {s[(s == group)].shape[0]}"
            )
            if self.strategy == "oversample":
                # Obtain indexes to resample
                sample_indexes = (
                    s[(s == group) & (y == label)]
                    .sample(n=size, replace=True, random_state=self.seed)
                    .index
                )

                sampled_X = X.loc[sample_indexes]
                sampled_y = y.loc[sample_indexes]
                sampled_s = s.loc[sample_indexes]

                final_X = pd.concat([final_X, sampled_X])
                final_y = pd.concat([final_y, sampled_y])
                final_s = pd.concat([final_s, sampled_s])

            else:  # Strategy = undersampling
                sample_indexes = (
                    s[(s == group) & (y == label)]
                    .sample(n=size, replace=False, random_state=self.seed)
                    .index
                )

                final_X = final_X.drop(sample_indexes)
                final_y = final_y.drop(sample_indexes)
                final_s = final_s.drop(sample_indexes)

            self.logger.debug(
                f"Final group {group} size: {final_s[(final_s == group)].shape[0]}"
            )

        self.logger.info("Data transformed.")
        return final_X.copy(), final_y.copy(), final_s.copy()

    def calculate_sample_sizes(
        self, p: int, n: int, target: float, strategy: str, alpha: float
    ) -> tuple[int, int]:
        """Calculate sample sizes, for both under and oversampling.

        Parameters
        ----------
        p : int
            Number of positive instances in a group.
        n : int
            Number of negative instances in a group.
        target : float
            Target prevalence for the group.
        strategy : str
            Strategy to obtain the desired prevalence.
            Either "undersample" or "oversample".
        alpha : float
            Parameter that controls the new prevalence values. A value of 0 keeps
            original prevalence, and a value of 1 equalizes the prevalence.

        Returns
        -------
        tuple(int, int)
            The label to sample (negative or positive) and the number of samples
            to remove (undersampling) or add (oversampling).
        """
        prevalence = p / (p + n)
        self.logger.debug(f"Calculated prevalence: {prevalence}")
        target = target * alpha + prevalence * (1 - alpha)
        self.logger.debug(f"Target prevalence: {target}")
        if strategy == "oversample":
            if prevalence < target:
                return 1, round((-p * target - n * target + p) / (target - 1))
            else:
                return 0, round(p * ((1 / target) - 1) - n)
        elif strategy == "undersample":
            if prevalence < target:
                return 0, round((p * (1 - 1 / target)) + n)
            else:
                return 1, round((p * target + n * target - p) / (target - 1))
        else:
            raise ValueError(f"Invalid strategy value. Try one of {STRATEGIES}.")
