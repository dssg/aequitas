from typing import Optional, Literal, Union
import warnings

import pandas as pd

from .bias import Bias
from .group import Group
from .plot import summary, disparity, absolute


class Audit:
    def __init__(
        self,
        df: pd.DataFrame,
        threshold: Optional[dict] = None,
        score_column: Optional[str] = "score",
        label_column: Optional[str] = "label",
        sensitive_attribute_column: Optional[Union[str, list[str]]] = None,
        reference_groups: Optional[Union[Literal["maj", "min"], dict]] = "maj",
    ):
        """
        This class allows to audit a model for fairness, and plot the results. The Audit
        class is a wrapper around the Group and Bias classes.

        It additionally allows to obtain global metrics of performance of your
        predictions or scores.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the scores, labels, and sensitive attribute(s).
        threshold : dict, optional
            Dictionary containing the threshold for the scores. If the scores are
            already binarized, this parameter is ignored. The keys of the dictionary are
            'threshold_type' and 'threshold_value'. These are used to construct an
            aequitas.flow.methods.postprocessing.Threshold object.
        score_column : str, optional
            Name of the column containing the scores (or predictions). By default,
            'score'.
        label_column : str, optional
            Name of the column containing the labels. By default, 'label'.
        sensitive_attribute_column : Union[str, list[str]], optional
            Name of the column(s) containing the sensitive attribute(s). If None, all
            columns except the score and label columns are considered sensitive
            attributes. By default, None.
        reference_groups : Union[Literal["maj", "min"],  dict], optional
            Reference group(s) to use for the bias metrics. If 'maj', the majority group
            is used. If 'min', the minority group is used. If a dictionary is passed,
            the keys are the sensitive attribute columns and the values are the
            reference groups. By default, 'maj'.
        """
        self.df = df
        self.score_column = score_column
        self.threshold = threshold
        self.label_column = label_column
        self.sensitive_attribute_column = sensitive_attribute_column
        self.binarized = False
        # Validate if the score column is in the DataFrame
        self._validate_score_column()
        # Validate if the label column is in the DataFrame
        self._validate_label_column()
        # Validate the sensitive attribute column(s) (if passed) is in the DataFrame
        self._validate_sensitive_attribute_column()
        self.label_column = label_column
        self.sensitive_attribute_column = sensitive_attribute_column
        self.reference_groups = reference_groups

    @classmethod
    def from_series(
        cls,
        scores: pd.Series,
        labels: pd.Series,
        sensitive_attributes: Union[pd.Series, pd.DataFrame],
        **kwargs,
    ):
        """
        Create an Audit object instance from Series of Scores, Labels, and Sensitive
        Attribute(s).

        Parameters
        ----------
        scores : pd.Series
            Column representing the scores of the model.
        labels : pd.Series
            Column representing the ground-truth labels.
        sensitive_attributes : Union[pd.Series, pd.DataFrame]
            Column or Columns representing the sensitive information of each instance.
        **kwargs
            Additional arguments to pass to the Audit class. Refer to the constructor
            for more information.

        Returns
        -------
        Audit
            Audit object instance.
        """
        df = pd.concat([scores, labels, sensitive_attributes], axis=1)
        return cls(df, **kwargs)

    def audit(self, group_args: Optional[dict] = {}, bias_args: Optional[dict] = {}):
        """
        Audit the model for fairness.

        Parameters
        ----------
        group_args : dict, optional
            Additional arguments to pass to the Group class. Refer to the documentation
            of the Group class for more information.
        bias_args : dict, optional
            Additional arguments to pass to the Bias class. Refer to the documentation
            of the Bias class for more information.
        """
        self.group = Group()
        self.metrics_df, _ = self.group.get_crosstabs(
            df=self.df,
            score_thresholds=None,
            attr_cols=self.sensitive_attribute_column,
            score_col=self.score_column,
            label_col=self.label_column,
            **group_args,
        )

        self.bias = Bias()
        if self.reference_groups == "maj":
            self.disparity_df = self.bias.get_disparity_major_group(
                self.metrics_df,
                self.df,
                **bias_args,
            )
        elif self.reference_groups == "min":
            self.disparity_df = self.bias.get_disparity_min_metric(
                self.metrics_df, self.df, **bias_args
            )
        else:
            self.disparity_df = self.bias.get_disparity_predefined_groups(
                self.metrics_df, self.df, self.reference_groups, **bias_args
            )

    def performance(self):
        """
        Return the performance of the model.

        Returns
        -------
        pd.DataFrame
            Performance metrics.
        """
        # Create a column in DF with a constant value for group
        self.df["group_performance"] = "all"
        g = Group()
        self.performance_metrics_df, _ = g.get_crosstabs(
            df=self.df,
            score_thresholds=None,
            attr_cols=["group_performance"],
            score_col=self.score_column,
            label_col=self.label_column,
        )
        self.df.drop(columns=["group_performance"], inplace=True)
        return self.performance_metrics_df

    def summary_plot(self, metrics, **kwargs):
        """
        Plot the summary of the audit.

        Parameters
        ----------
        metrics : str or list[str]
            Metrics to plot. Refer to the documentation of the summary function for
            more information.
        **kwargs
            Additional arguments to pass to the summary function. Refer to the
            documentation of the summary function for more information.

        Returns
        -------
        matplotlib.pyplot.Axes
            Axes containing the summary plot.
        """
        if not isinstance(metrics, list):
            metrics = [metrics]
        return summary(self.disparity_df, metrics, **kwargs)

    def disparity_plot(self, metrics, attribute, **kwargs):
        """
        Plot the disparity of the audit.

        Parameters
        ----------
        metrics : str or list[str]
            Metrics to plot. Refer to the documentation of the disparity function for
            more information.
        attribute : str
            Name of the sensitive attribute to plot.
        **kwargs
            Additional arguments to pass to the disparity function. Refer to the
            documentation of the disparity function for more information.

        Returns
        -------
        matplotlib.pyplot.Axes
            Axes containing the disparity plot.
        """
        if not isinstance(metrics, list):
            metrics = [metrics]
        return disparity(self.disparity_df, metrics, attribute, **kwargs)

    def absolute_plot(self, metrics, attribute, **kwargs):
        """
        Plot the absolute disparity of the audit.

        Parameters
        ----------
        metrics : str or list[str]
            Metrics to plot. Refer to the documentation of the absolute function for
            more information.
        attribute : str
            Name of the sensitive attribute to plot.
        **kwargs
            Additional arguments to pass to the absolute function. Refer to the
            documentation of the absolute function for more information.

        Returns
        -------
        matplotlib.pyplot.Axes
            Axes containing the absolute disparity plot.
        """
        if not isinstance(metrics, list):
            metrics = [metrics]
        return absolute(self.disparity_df, metrics, attribute, **kwargs)

    def _validate_score_column(self):
        # Check if column exists
        if not (self.score_column in self.df.columns):
            raise ValueError(
                f"Score column {self.score_column} not found in the DataFrame."
            )
        # Check if values are numeric
        if not pd.api.types.is_numeric_dtype(self.df[self.score_column]):
            raise ValueError(f"Score column {self.score_column} must be numeric.")
        # Check if values are either 0 or 1 (already binarized)
        if self.df[self.score_column].isin([0, 1]).all():
            self.binarized = True
        # If binarized and a threshold is passed, warn the user that the threshold
        # will be ignored
        if self.binarized and self.threshold is not None:
            warnings.warn("Scores are already binarized. Thresholds will be ignored.")
        # If not binarized and a threshold is not passed, raise an error
        if not self.binarized and self.threshold is None:
            raise ValueError("Scores are not binarized. Please pass a threshold.")

    def _validate_label_column(self):
        # Check if column exists
        if not (self.label_column in self.df.columns):
            raise ValueError(
                f"Label column {self.label_column} not found in the DataFrame."
            )
        # If not binarized, raise an error
        if not self.df[self.label_column].isin([0, 1]).all():
            raise ValueError(f"Label column {self.label_column} must be binarized.")

    def _validate_sensitive_attribute_column(self):
        # If column is None, check if there are more columns than the score and label
        if self.sensitive_attribute_column is None:
            sensitive_attribute_columns = self.df.columns.drop(
                [self.score_column, self.label_column]
            )
            if len(sensitive_attribute_columns) == 0:
                raise ValueError(
                    "Sensitive attribute column(s) not found in the DataFrame."
                )
            self.sensitive_attribute_column = sensitive_attribute_columns
        # If column is a string, check if it exists
        if isinstance(self.sensitive_attribute_column, str):
            if not (self.sensitive_attribute_column in self.df.columns):
                raise ValueError(
                    f"Sensitive attribute column {self.sensitive_attribute_column} not"
                    " found in the DataFrame."
                )
        # If column is a list, check if all columns exist
        if isinstance(self.sensitive_attribute_column, list):
            if not all(
                [col in self.df.columns for col in self.sensitive_attribute_column]
            ):
                raise ValueError(
                    f"Sensitive attribute column(s) {self.sensitive_attribute_column}"
                    " not found in the DataFrame."
                )
        # Check if values are categorical / object
        for dtype in self.df[self.sensitive_attribute_column].dtypes:
            if dtype != "object":
                raise ValueError(
                    f"Sensitive attribute column(s) {self.sensitive_attribute_column}"
                    " must be categorical."
                )

    @property
    def confusion_matrix(self) -> pd.DataFrame:
        """
        Return the confusion matrix of the model.
        """
        absolute_metrics = self.group.list_absolute_metrics(self.metrics_df)
        remove_columns = ["model_id", "score_threshold", "k"] + absolute_metrics

        return self.metrics_df.drop(columns=remove_columns).set_index(
            ["attribute_name", "attribute_value"]
        )

    @property
    def metrics(self) -> pd.DataFrame:
        """
        Return the confusion matrix metrics of the model.
        """
        absolute_metrics = self.group.list_absolute_metrics(self.metrics_df)
        include_columns = ["attribute_name", "attribute_value"] + absolute_metrics

        return self.metrics_df[include_columns].set_index(
            ["attribute_name", "attribute_value"]
        )

    @property
    def disparities(self) -> pd.DataFrame:
        """
        Return the disparities of the model.
        """
        disparity_metrics = self.bias.list_disparities(self.disparity_df)

        try:
            # Check if there are statistical tests in df
            significance_metrics = self.bias.list_significance(self.disparity_df)
            disparity_metrics += significance_metrics
        except KeyError:
            pass

        include_columns = ["attribute_name", "attribute_value"] + disparity_metrics

        return self.disparity_df[include_columns].set_index(
            ["attribute_name", "attribute_value"]
        )
