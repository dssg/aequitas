import logging
import altair as alt

from aequitas.plot.commons import validators as Validator
from aequitas.plot.commons import scales as Scales
from aequitas.plot.commons.helpers import to_list, calculate_chart_size_from_elements


def __sanitize_metrics(metrics_list):
    try:
        return [metric.lower() for metric in to_list(metrics_list)]
    except AttributeError:
        logging.error('"metrics_list" must be a list of strings.')


def __filter_df(disparity_df, metrics, attribute, xy_plot=False):
    viz_fields = [
        "attribute_name",
        "attribute_value",
        "group_size",
        "total_entities",
    ]
    viz_fields += metrics

    if not xy_plot:
        viz_fields += [f"{metric}_disparity" for metric in metrics]

    plot_table = disparity_df[viz_fields][
        disparity_df["attribute_name"] == attribute
    ].copy(deep=True)

    return plot_table


def __validate_data_inputs(disparity_df, metrics, attributes, fairness_threshold):
    Validator.metrics(metrics)
    Validator.dataframe(disparity_df, metrics)
    Validator.attributes(disparity_df, attributes)
    Validator.fairness_threshold(fairness_threshold)


def __calculate_default_sizes(
    chart_height, chart_width, chart_default_sizes, metrics=None, attributes=None
):
    if chart_height is None:
        num_metrics = len(metrics)

        chart_height = calculate_chart_size_from_elements(
            chart_default_sizes.vertical_header_height,
            chart_default_sizes.line_height,
            num_metrics,
        )

    if chart_width is None:
        num_attributes = len(attributes)

        chart_width = calculate_chart_size_from_elements(
            chart_default_sizes.horizontal_header_width,
            chart_default_sizes.column_width,
            num_attributes,
        )

    return (chart_height, chart_width)


def __get_global_scales(
    plot_table, ref_group, metrics, chart_height, accessibility_mode
):
    global_scales = dict()
    global_scales["color"] = Scales.get_color_scale(plot_table, ref_group)
    global_scales["bubble_size"] = Scales.get_bubble_size_scale(
        plot_table, metrics, chart_height
    )
    global_scales["shape"] = Scales.get_shape_scale(
        plot_table, ref_group, accessibility_mode
    )

    return global_scales


def prepare_bubble_chart(
    disparity_df,
    metrics_list,
    attribute,
    fairness_threshold,
    chart_height,
    chart_width,
    chart_default_sizes,
    accessibility_mode,
):
    metrics = __sanitize_metrics(metrics_list)

    __validate_data_inputs(disparity_df, metrics, attribute, fairness_threshold)

    ## Calculate default chart sizes
    num_metrics = len(metrics)

    (chart_height, chart_width) = __calculate_default_sizes(
        chart_height, chart_width, chart_default_sizes, metrics
    )

    Validator.chart_size_bubble(
        chart_width, chart_height, chart_default_sizes, num_metrics
    )

    # GET REF GROUP
    ref_group = disparity_df.loc[disparity_df["attribute_name"] == attribute][
        f"{metrics[0]}_ref_group_value"
    ].iloc[0]

    plot_table = __filter_df(disparity_df, metrics, attribute)

    # GLOBAL CHART MECHANICS
    ## SCALES: COLOR, BUBBLE SIZE, SHAPE
    global_scales = __get_global_scales(
        plot_table, ref_group, metrics, chart_height, accessibility_mode
    )

    # SELECTION
    selection = alt.selection_multi(empty="all", fields=["attribute_value"])

    return (
        plot_table,
        metrics,
        ref_group,
        global_scales,
        chart_height,
        chart_width,
        selection,
    )


def prepare_summary_chart(
    disparity_df,
    metrics_list,
    attributes_list,
    fairness_threshold,
    chart_height,
    chart_width,
    chart_default_sizes,
):
    ## If a specific list of attributes was not passed, use all from df
    if fairness_threshold is None:
        raise ValueError("Fairness threshold cannot be None for the Summary chart.")

    metrics = __sanitize_metrics(metrics_list)

    if attributes_list is None:
        attributes = list(disparity_df["attribute_name"].unique())
    else:
        attributes = to_list(attributes_list)

    __validate_data_inputs(disparity_df, metrics, attributes, fairness_threshold)

    # CHART SIZES
    (chart_height, chart_width) = __calculate_default_sizes(
        chart_height, chart_width, chart_default_sizes, metrics, attributes
    )

    return (metrics, attributes, chart_height, chart_width)


def prepare_xy_chart(
    disparity_df,
    x_metric,
    y_metric,
    attribute,
    fairness_threshold,
    chart_height,
    chart_width,
    chart_default_sizes,
    accessibility_mode,
):
    metrics = __sanitize_metrics([x_metric, y_metric])

    x_metric = metrics[0]
    y_metric = metrics[1]

    __validate_data_inputs(disparity_df, metrics, attribute, fairness_threshold)

    if (chart_width is None) and (chart_height is None):
        chart_width = chart_default_sizes.full_side
        chart_height = chart_default_sizes.full_side
    elif chart_width is None:
        chart_width = chart_height
    elif chart_height is None:
        chart_height = chart_width

    Validator.chart_size_xy(chart_width, chart_height)

    # REF GROUP DEFINITION
    ## ref_group is taken from x_metric as long as it's not part of the metrics that
    ## do not have disparity variables (such as prev)
    if x_metric in Validator.NON_DISPARITY_METRICS_LIST:
        ref_group = disparity_df[disparity_df["attribute_name"] == attribute].iloc[0][
            f"{y_metric}_ref_group_value"
        ]
    else:
        ref_group = disparity_df[disparity_df["attribute_name"] == attribute].iloc[0][
            f"{x_metric}_ref_group_value"
        ]

    plot_table = __filter_df(disparity_df, metrics, attribute, xy_plot=True)

    # SCALES
    global_scales = __get_global_scales(
        plot_table, ref_group, metrics, chart_height, accessibility_mode
    )

    # SELECTION
    interactive_selection_group = alt.selection_multi(
        empty="all", fields=["attribute_value"]
    )

    return (
        plot_table,
        x_metric,
        y_metric,
        ref_group,
        global_scales,
        chart_height,
        chart_width,
        interactive_selection_group,
    )
