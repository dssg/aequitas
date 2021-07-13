import logging
from aequitas.plot.commons.style import sizes as Sizes
from aequitas.plot.commons.helpers import calculate_chart_size_from_elements, to_list

METRICS_LIST = [
    "tpr",
    "tnr",
    "for",
    "fdr",
    "fpr",
    "fnr",
    "npv",
    "ppr",
    "precision",
    "pprev",
    "prev",
    "group_size_pct",
]
NON_DISPARITY_METRICS_LIST = ["prev", "group_size_pct"]
DF_FIELDS = ["attribute_name", "attribute_value", "group_size", "total_entities"]

# DATA


def dataframe(df, metrics_list):
    """Validates if dataframe has all the necessary fieds to plot charts.

    :param user_input_df: a dataframe from Aequitas
    :type user_input_df: pandas.core.frame.DataFrame
    :param metrics_list: a list of valid metrics
    :type metrics_list: list
    """
    for field in DF_FIELDS:
        try:
            df[field]
        except KeyError as e:
            logging.error(f'The dataframe does not contain the column "{field}"')
            raise e

    for metric in metrics_list:
        try:
            df[metric]
        except KeyError as e:
            logging.error(f'The dataframe does not contain the column "{metric}".')
            raise e
        if metric not in NON_DISPARITY_METRICS_LIST:
            try:
                df[f"{metric}_disparity"]
            except KeyError as e:
                logging.error(
                    f'The dataframe does not contain the column "{metric}_disparity".'
                )
                raise e


def metrics(metrics):
    """Validates if  if there are no duplicated metrics, and if all metrics are valid fairness metrics.

    :param metrics: a metric or a list of metrics
    :type metrics: str or list 

    """

    metrics = to_list(metrics)

    if len(metrics) != len(set(metrics)):
        raise ValueError(
            f"There is at least one duplicated metric in the metrics list."
        )

    for metric in metrics:
        if metric not in METRICS_LIST:
            raise ValueError(f"{metric} is not a valid metric.")


def attributes(df, attributes):
    """Validates if attributes exists in the dataframe.

    :param df: a dataframe from Aequitas
    :type df: pandas.core.frame.DataFrame
    :param attributes: an attribute or a list of attributes
    :type attributes: str or list 
    """

    attributes = to_list(attributes)

    for attribute in attributes:
        attribute_df = df[df["attribute_name"] == attribute]

        if len(attribute_df) == 0:
            raise ValueError(
                f"Attribute name {attribute} does not exist in the dataframe."
            )


def fairness_threshold(fairness_threshold):
    """Checks if fairness_threshold is a number greater than 1.
    :param fairness_threshold: a threshold value
    :type fairness_threshold: number
    """
    if fairness_threshold is not None:
        try:
            if fairness_threshold <= 1:
                raise ValueError(f"Fairness Threshold value must be greater than 1.")
        except TypeError as e:
            logging.error("Fairness Threshold value must be a number")
            raise e


# SIZES


def chart_size_summary(size_constants, num_metrics, num_attributes):
    """Given sizing constants for the summary chart and the number of metrics and attributes, this function checks if there is enough area to render all elements and if there is a sensible aspect ratio.

    :param size_constants: size constants for Summary Chart size elements
    :type size_constants: dict
    :param num_metrics: number of metrics to be plotted (rows)
    :type num_metrics: int
    :param num_attributes: number of attributes to be plotted (columns)
    :type num_attributes: int
    """

    min_width = (
        Sizes.Summary_Chart.min_column_width
        * num_attributes
        / Sizes.Summary_Chart.column_width_ratio
    )
    max_width = (
        Sizes.Summary_Chart.max_column_width
        * num_attributes
        / Sizes.Summary_Chart.column_width_ratio
    )
    min_height = Sizes.Summary_Chart.min_line_height * num_metrics / 0.65
    max_height = Sizes.Summary_Chart.max_line_height * num_metrics / 0.65

    if size_constants["column_width"] < Sizes.Summary_Chart.min_column_width:
        raise ValueError(
            f"The chart width must be at least {min_width} to accommodate the {num_attributes} attributes to be plotted."
        )

    if size_constants["column_width"] > Sizes.Summary_Chart.max_column_width:
        raise ValueError(
            f"The chart width must be smaller than {max_width} to preserve a sensible aspect ratio."
        )
    if size_constants["line_height"] < Sizes.Summary_Chart.min_line_height:
        raise ValueError(
            f"The chart height must be at least {min_height} to accommodate the {num_metrics} metrics to be plotted."
        )
    if size_constants["line_height"] > Sizes.Summary_Chart.max_line_height:
        raise ValueError(
            f"The chart height must be smaller than {max_height} to preserve a sensible aspect ratio."
        )


def chart_size_bubble(width, height, chart_sizes, num_metrics):
    """Given parameters width and height for the bubble charts and the number of metrics, this function checks if there is enough area to render all elements and if there is a sensible aspect ratio.

    :param width: chart width
    :type width: int
    :param height: chart height
    :type height: int
    :param chart_sizes: size style defaults for the chart 
    :type chart_sizes: class
    :param num_metrics: number of metrics to be plotted
    :type num_metrics: int
    """
    min_height = calculate_chart_size_from_elements(
        chart_sizes.vertical_header_height, chart_sizes.min_line_height, num_metrics,
    )

    if width < chart_sizes.min_width:
        raise ValueError(
            f"The chart width must be at least {chart_sizes.min_width} to preserve a sensible aspect ratio."
        )

    if height < min_height:
        raise ValueError(
            f"The chart height must be at least {min_height} to accommodate the {num_metrics} metrics to be plotted."
        )


def chart_size_xy(width, height):
    """Given parameters width and height for the XY chart, this function checks if height is equal to width and if there is enough area to render all elements.

    :param width: chart width
    :type width: int
    :param height: chart height
    :type height: int
    """
    if width != height:
        raise ValueError(f"The chart width and height must be the same.")

    if width < Sizes.XY_Chart.min_side:
        raise ValueError(
            f"The chart width must be at least {Sizes.XY_Chart.min_side} to preserve a sensible aspect ratio."
        )

