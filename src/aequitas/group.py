import logging
import uuid
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logging.getLogger(__name__)

__author__ = (
    "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Benedict Kuester, Loren Hinkson, Sérgio Jesus"
)


class Group(object):
    """
    Calculates absolute metrics for a given dataframe with model scores and
    labels, divided by different values on protected attributes.

    Attributes
    ----------
    absolute_metrics : List[str]
        Calculated bias metrics.

    Methods
    -------
    get_crosstabs(df, score_thresholds=None, attr_cols=None, score_col="score", label_col="label_value")
        Calculates metrics for a given dataframe.
    get_multimodel_crosstabs(df, score_thresholds=None, attr_cols=None, score_col="score", label_col="label_value")
        Calculates metrics for a given dataframe with multiple models.
    """

    all_group_metrics = (
        "ppr",
        "pprev",
        "precision",
        "fdr",
        "for",
        "fpr",
        "fnr",
        "tpr",
        "tnr",
        "npv",
        "prev",
    )

    def __init__(self, input_group_metrics=all_group_metrics):
        self.absolute_metrics = input_group_metrics

    @staticmethod
    def gen_df_from_confusion_matrix(
            tp: int,
            tn: int,
            fp: int,
            fn: int,
            k: int,
            model_id: str,
            score_threshold: str,
            attribute_name: str,
            attribute_value: str,
            total_entities: int,
    ):
        columns = [  # Columns in the resulting dataframe
            "model_id",
            "score_threshold",
            "k",
            "attribute_name",
            "attribute_value",
            "tpr",
            "tnr",
            "fomr",
            "fdr",
            "fpr",
            "fnr",
            "npv",
            "precision",
            "pp",
            "pn",
            "ppr",
            "pprev",
            "fp",
            "fn",
            "tn",
            "tp",
            "group_label_pos",
            "group_label_neg",
            "group_size",
            "total_entities",
            "prev",
        ]
        final_dict = {column: [] for column in columns}
        divide = lambda x, y: x / y if y != 0 else np.nan
        # Calculate all metrics from confusion matrix.
        tpr = divide(tp, fn + tp)
        tnr = divide(tn, fp + tn)
        # We can't have variables named 'for', this is changed in return
        fomr = divide(fn, fn + tn)
        fdr = divide(fp, tp + fp)
        fpr = divide(fp, fp + tn)
        fnr = divide(fn, fn + tp)
        npv = divide(tn, fn + tn)
        precision = divide(tp, tp + fp)
        pp = fp + tp
        pn = fn + tn
        ppr = divide(pp, k)
        pprev = divide(pp, pp + pn)
        group_label_pos = fn + tp
        group_label_neg = fp + tn
        group_size = group_label_pos + group_label_neg
        prev = (fn + tp) / group_size
        for key in final_dict.keys():
            # Maybe using locals is not the best, but it is the least
            # messy solution in terms of code.
            final_dict[key].append(locals()[key])
        # Rename column fomr -> for (impossible to have variable named for).
        return pd.DataFrame(final_dict).rename(columns={"fomr": "for"})

    @staticmethod
    def gen_metrics_df(
        df: pd.DataFrame,
        attr_cols: List[str],
        score: str,
        label: str,
        score_threshold: str,
    ) -> pd.DataFrame:
        """
        Generates dataframe with metrics for a given threshold.

        Generates the confusion matrix given the scores in the `score` column
        and the "label_value" for each unique value in each of the `attr_cols`.
        Then, the metrics that are derived from the confusion matrix are
        calculated.

        Parameters
        ----------
        df : pandas.DataFrame
            Classification, label and protected attributes on a given dataset.
            The classification column must already be binary, and in
            column `score`. The label column must be in the column
            "label_value".
        attr_cols : List[str]
            Columns of the dataframe that represent protected attributes.
        score : str
            Column of the dataframe that represents classification result.
            Values in the column must be binary.
        label : str
            Column of the dataframe that represents label of instance.
            Values in the column must be binary.
        score_threshold : str
            Type of threshold used to binarize score.

        Returns
        -------
        pandas.DataFrame
            Metrics for each group in the protected attribute.
        """
        # Method to handle divisions by 0.
        model_id = Group._check_model_id(df, "df")
        # k = Total number of predicted positives in sample.
        # This is for pd.DataFrames only.
        k = df[df[score] == 1].shape[0]
        metrics_dfs = []
        for attribute_name in attr_cols:
            # Create confusion matrix for each group in the form of dictionary.
            # This is for pd.DataFrames only.
            grouped_data = (
                df.groupby(by=[attribute_name, score, label]).size().to_dict()
            )
            # Select the possible values of the protected attributes (groups)
            attribute_values = set([key[0] for key in grouped_data.keys()])
            for attribute_value in attribute_values:
                # Get confusion matrix from the dictionary.
                tp = grouped_data.get((attribute_value, 1, 1), 0)
                tn = grouped_data.get((attribute_value, 0, 0), 0)
                fp = grouped_data.get((attribute_value, 1, 0), 0)
                fn = grouped_data.get((attribute_value, 0, 1), 0)
                total_entities = df.shape[0]
                metrics_df = Group.gen_df_from_confusion_matrix(
                    tp, tn, fp, fn, k, model_id, score_threshold, attribute_name,
                    attribute_value, total_entities,
                )
                metrics_dfs.append(metrics_df)
        return pd.concat(metrics_dfs).reset_index().drop(columns="index")

    @staticmethod
    def _check_model_id(df: pd.DataFrame, method_table_name: str = "df") -> int:
        """
        Returns the model id in a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Predictions of a model. Might have or not the column "model_id".
        method_table_name: str
            Reference name of the dataframe for error message.

        Returns
        -------
        int
            The model id. 0 if no model id column exists.
        """
        if "model_id" in df.columns:
            df_models = df.model_id.unique()
            if len(df_models) != 1:
                raise ValueError(
                    "This method requires one and only one model_id in the "
                    "dataframe. "
                    f"Tip: Check that {method_table_name}.model_id.unique() "
                    "only returns an one-element array. "
                )
            else:
                return df_models[0]
        else:
            return 0

    def get_multimodel_crosstabs(
        self,
        df: pd.DataFrame,
        score_thresholds: Dict[str, List[Union[int, float]]] = None,
        attr_cols: List[str] = None,
        score_col: str = "score",
        label_col: str = "label_value",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Performs `get_crosstabs()` and calculates group metrics for results
        from multiple models.

        Parameters
        ----------
        df: pandas.DataFrame
            Results of model classification. Must contain:
            [score_col,  label_col, "model_id"].
        score_thresholds: Dict[str, List[Union[int, float]]]
            The possible thresholds of the models.
            Accepted keys:
                - 'rank_abs' for absolute ranking (top k scores classified as
                   positive).
                - 'rank_pct' for percentage ranking (top X% classified as
                   positive).
                - 'score_val' for value threshold (Scores above Y classified as
                   positive).
            The values for each key represent the different tested thresholds.
            These must be inside a list.
            If not specified, it is assumed the scores are already in binary
            form.
        attr_cols: List[str]
            Columns with protected attributes. If not specified, all
            columns except 'id', 'model_id', 'entity_id', 'score', 'label_value'
            are used. The values in these columns must be categorical.
        score_col : str
            Column of the dataframe that represents classification result.
            Must be numeric / binary.
        label_col : str
            Column of the dataframe that represents label of instance.
            Values in the column must be binary.
        Returns
        -------
        pandas.DataFrame
            Metrics for each group in the protected attribute, for combination
            of model and threshold. List of protected attributes.
        """
        if "model_id" not in df.columns:
            raise ValueError(
                'The method expects a column named "model_id" in the ' "dataframe."
            )

        df_models = df.model_id.unique()
        crosstab_list = []
        model_attr_cols = None
        for model in df_models:
            model_df = df.loc[df["model_id"] == model]
            model_crosstab, model_attr_cols = self.get_crosstabs(
                model_df,
                score_thresholds=score_thresholds,
                attr_cols=attr_cols,
                score_col=score_col,
                label_col=label_col,
            )
            crosstab_list.append(model_crosstab)
        # Note: only returns model_attr_cols from the last iteration, as they
        # all are the same
        return pd.concat(crosstab_list, ignore_index=True), model_attr_cols

    def get_crosstabs(
        self,
        df,
        score_thresholds: Dict[str, List[Union[int, float]]] = None,
        attr_cols: List[str] = None,
        score_col: str = "score",
        label_col: str = "label_value",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
         Creates univariate groups and calculates group metrics for results
         from a single model.

         Parameters
         ----------
         df : pandas.DataFrame
             Results of model classification. Must contain:
             [score_column,  label_column].
         score_thresholds : Dict[str, List[Union[int, float]]]
             The possible thresholds of the model.
             Accepted keys:
                 - 'rank_abs' for absolute ranking (top k scores classified as
                    positive).
                 - 'rank_pct' for percentage ranking (top X% classified as
                    positive).
                 - 'score_val' for value threshold (Scores above Y classified as
                    positive).
             The values for each key represent the different tested thresholds.
             These must be inside a list.
             If not specified, it is assumed the scores are already in binary
             form.
         attr_cols: List[str]
             Columns with protected attributes. If not specified, all
             columns except 'id', 'model_id', 'entity_id', 'score', 'label_value'
             are used. The values in these columns must be categorical.
         score_col : str
             Column of the dataframe that represents classification result.
             Must be numeric / binary.
         label_col : str
             Column of the dataframe that represents label of instance.
             Values in the column must be binary.
         Returns
         -------
        Tuple[pd.DataFrame, List[str]]
             Metrics for each group in the protected attribute, each specified
             threshold. List of protected attributes.
        """
        if not attr_cols:
            non_attr_cols = ["id", "model_id", "entity_id", score_col, label_col]
            # index of the columns that are protected attributes.
            attr_cols = df.columns[~df.columns.isin(non_attr_cols)]

        necessary_cols = list(attr_cols) + [score_col, label_col]
        for col in ["id", "model_id", "entity_id"]:
            if col in df.columns:
                necessary_cols.append(col)
        # Copy only the necessary columns
        df = df[necessary_cols].copy()  # To not transform original dataframe.

        # Validation step.
        for col in [score_col, label_col]:
            # Validate if column is in dataframe.
            if col not in df.columns:
                raise KeyError(f'The column "{col}" is not on the dataframe.')
        if len(df[label_col].unique()) > 2:
            raise ValueError("Labels are not binarized.")
        else:
            df[label_col] = df[label_col].astype(int)

        for col in attr_cols:
            # Validate if column is in dataframe.
            if col not in df.columns:
                raise KeyError(f'The attribute column "{col}" is not on the dataframe.')
            # Validate if column has correct datatype.
            if df[col].dtype not in (object, str):
                raise TypeError(
                    f'The attribute column "{col}" has in invalid datatype.'
                )

        # In case no threshold is provided.
        metrics_matrices = []
        if not score_thresholds and len(df[score_col].unique()) > 2:
            raise ValueError("Scores are not binarized. Provide a threshold.")
        elif not score_thresholds:  # No thresholds given and binarized
            df[score_col] = df[score_col].astype(int)
            metrics_matrices.append(
                self.gen_metrics_df(df, attr_cols, score_col, label_col, "binary 0/1")
            )
            score_thresholds = {}
        # In case a threshold is provided.
        binarized_column = str(uuid.uuid4())  # Add a column with binary scores.
        # This is done to maintain the score for future iterations in threshold.
        sorted_df = False  # Flag to sort the dataframe in the first run.
        for key, values in score_thresholds.items():
            for value in values:
                if not sorted_df:
                    # Sort df only once, at the first iteration.
                    df = df.sort_values(score_col, ascending=False).reset_index(
                        drop=True
                    )
                    sorted_df = True
                # Cast scores as ints in the dataframe
                if key == "rank_abs":
                    df[binarized_column] = (df.index < value).astype(int)
                elif key == "rank_pct":
                    df[binarized_column] = (df.index < value * df.shape[0]).astype(int)
                elif key == "score_val":
                    df[binarized_column] = (df[score_col] >= value).astype(int)
                else:
                    raise KeyError("Invalid keys")
                metrics_matrices.append(
                    self.gen_metrics_df(
                        df,
                        attr_cols,
                        binarized_column,
                        label_col,
                        f"{value}_{key[-3:]}",
                    )
                )
        return (pd.concat(metrics_matrices).reset_index().drop(columns="index"),
                attr_cols)

    def list_absolute_metrics(self, df: pd.DataFrame) -> List[Any]:
        """
        View list of all calculated absolute bias metrics in df.

        Parameters
        ----------
        df : pandas.DataFrame
            The result of the `get_crosstabs` methods.

        Returns
        -------
        List[Any]
            Absolute bias metrics.
        """
        return df.columns.intersection(self.absolute_metrics).tolist()
