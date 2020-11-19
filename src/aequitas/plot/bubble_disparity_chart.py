import math
import altair as alt
import pandas as pd

from aequitas.plot.commons.helpers import (
    no_axis,
    transform_ratio,
)
from aequitas.plot.commons.legend import draw_legend
from aequitas.plot.commons.scales import get_chart_size_range
from aequitas.plot.commons.tooltips import (
    get_tooltip_text_group_size,
    get_tooltip_text_disparity_explanation,
)
from aequitas.plot.commons.style.classes import (
    Metric_Axis,
    Axis,
    Annotation,
    Reference_Rule,
    Threshold_Rule,
    Threshold_Band,
    Bubble,
    Chart_Title,
)

from aequitas.plot.commons.style.text import FONT
from aequitas.plot.commons.style.sizes import Disparity_Chart
from aequitas.plot.commons import initializers as Initializer


# Altair 2.4.1 requires that all chart receive a dataframe, for charts that don't need it
# (like most annotations), we pass the following dummy dataframe to reduce the complexity of the resulting vega spec.
DUMMY_DF = pd.DataFrame({"a": [1, 1], "b": [0, 0]})


def __get_position_scales(
    plot_table, metrics, fairness_threshold, chart_height, chart_width
):
    """Computes the scales for x and y encodings to be used in the disparity bubble chart."""

    position_scales = dict()

    # DISPARITIES SCALE
    # RANGE
    x_range = get_chart_size_range(chart_width, Disparity_Chart.padding_x)

    # DOMAIN
    # Get max absolute disparity
    scaled_disparities_col_names = [f"{metric}_disparity_scaled" for metric in metrics]

    def max_column(x):
        return max(x.min(), x.max(), key=abs)

    max_disparities = plot_table[scaled_disparities_col_names].apply(max_column, axis=1)
    abs_max_disparity = abs(max_column(max_disparities))

    # If fairness_threshold is defined, get max between threshold and max absolute disparity
    if fairness_threshold is not None:
        x_domain_limit = math.ceil(max(abs_max_disparity, fairness_threshold))
    else:
        x_domain_limit = math.ceil(abs_max_disparity)

    x_domain = [-x_domain_limit, x_domain_limit]
    position_scales["x"] = alt.Scale(domain=x_domain, range=x_range)

    # METRICS SCALE
    y_range = get_chart_size_range(chart_height, Disparity_Chart.padding_y)
    if chart_height < 300:
        y_range[0] = 30
    y_domain = [metric.upper() for metric in metrics]
    position_scales["y"] = alt.Scale(domain=y_domain, range=y_range)

    return position_scales


def __draw_metrics_rules(metrics, scales, concat_chart):
    """Draws an horizontal rule and the left-hand side label for each metric.
    The groups' bubbles will be positioned on this horizontal rule."""

    metrics_labels = [metric.upper() for metric in metrics]

    metrics_axis = alt.Axis(
        domain=False,
        ticks=False,
        orient="left",
        labelAngle=Metric_Axis.label_angle,
        # LabelPadding logic:
        # Spaces the labels further from the chart if they are part of a concatenated chart
        labelPadding=Metric_Axis.label_padding
        if not concat_chart
        else Metric_Axis.label_padding_concat_chart,
        title="",
    )

    rules_df = pd.DataFrame(
        {
            "metric": metrics_labels,
            "x": scales["x"]["domain"][0],
            "x2": scales["x"]["domain"][1],
        }
    )

    metrics_rules = (
        alt.Chart(rules_df)
        .mark_rule(
            strokeWidth=Metric_Axis.stroke_width,
            stroke=Metric_Axis.stroke,
            tooltip="",
        )
        .encode(
            y=alt.Y("metric:N", scale=scales["y"], axis=metrics_axis),
            x=alt.X("x:Q", scale=scales["x"]),
            x2="x2:Q",
        )
    )

    return metrics_rules


def __get_x_axis_values(x_domain, zero=True):
    TICK_STEP_OPTIONS = [1, 2, 5, 10, 20, 50, 100]

    def list_axis_values(limit, step):
        axis_start = max([1, step - 1])
        positive_axis_values = list(range(axis_start, limit, step))
        negative_axis_values = [-x for x in positive_axis_values][::-1]
        axis_values = positive_axis_values + negative_axis_values
        return axis_values

    domain_limit = x_domain[1] + 1

    for tick_step in TICK_STEP_OPTIONS:
        if domain_limit / tick_step <= 6 or tick_step == TICK_STEP_OPTIONS[-1]:
            axis_values = list_axis_values(domain_limit, tick_step)
            break

    if zero:
        return axis_values + [0]
    return axis_values


def __draw_x_ticks_labels(scales, chart_height):
    """Draws the numbers in the horizontal axis."""

    # The values to be drawn, we don't want to draw 0 (which corresponds to a ratio of 1) as we later draw an annotation.
    axis_values = __get_x_axis_values(scales["x"].domain)

    # Given the semantic of the chart, (how many times smaller or larger) we draw absolute values.
    axis_values_labels = [abs(x) + 1 if x != 0 else "=" for x in axis_values]

    axis_df = pd.DataFrame({"value": axis_values, "label": axis_values_labels})

    tick_labels = (
        alt.Chart(axis_df)
        .mark_text(
            tooltip="",
            align="center",
            font=FONT,
            fontSize=Axis.label_font_size,
            fontWeight=Axis.label_font_weight,
            color=Axis.label_color,
        )
        .encode(
            text=alt.Text("label:N"),
            x=alt.X(
                "value:Q",
                scale=scales["x"],
            ),
            y=alt.value(Disparity_Chart.padding_y * chart_height * 0.7),
        )
    )

    return tick_labels


def __draw_text_annotations(ref_group, chart_height, x_range):
    """Draws on chart text annotations."""

    # FONT
    annotation_text_params = dict(
        font=FONT,
        fontWeight=Annotation.font_weight,
        tooltip="",
    )

    # TIMES LARGER TEXT
    text_times_larger = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="right",
            fill=Annotation.font_color,
            fontSize=Annotation.font_size,
            **annotation_text_params,
        )
        .encode(
            x=alt.value(x_range[1]),
            y=alt.value(Disparity_Chart.padding_y * chart_height * 0.3),
            text=alt.value("Times Larger"),
        )
    )

    # TIMES SMALLER TEXT
    text_times_smaller = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="left",
            fill=Annotation.font_color,
            fontSize=Annotation.font_size,
            **annotation_text_params,
        )
        .encode(
            x=alt.value(x_range[0]),
            y=alt.value(Disparity_Chart.padding_y * chart_height * 0.3),
            text=alt.value("Times Smaller"),
        )
    )

    # EQUAL TEXT
    text_reference_group = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="center",
            fontSize=Annotation.font_size,
            fill=Annotation.font_color,
            **annotation_text_params,
        )
        .encode(
            x=alt.value(x_range[0] + (x_range[1] - x_range[0]) / 2),
            y=alt.value(Disparity_Chart.padding_y * chart_height * 0.3),
            text=alt.value("Equal"),
        )
    )

    return text_times_smaller + text_times_larger + text_reference_group


def __draw_reference_rule(ref_group, chart_height, chart_width):
    """Draws vertical reference rule where the ratio is the same as the reference group."""

    reference_rule = (
        alt.Chart(DUMMY_DF)
        .mark_rule(
            strokeWidth=Reference_Rule.stroke_width,
            stroke=Reference_Rule.stroke,
            strokeDash=Reference_Rule.stroke_dash,
        )
        .encode(
            x=alt.value(chart_width / 2),
            y=alt.value(chart_height * Disparity_Chart.padding_y / 1.2),
            y2=alt.value(chart_height * (1 - Disparity_Chart.padding_y / 1.2)),
            tooltip=alt.value(f"{ref_group} [REF]"),
        )
    )

    return reference_rule


def __draw_threshold_rules(
    threshold_df, scales, chart_height, accessibility_mode=False
):
    """Draws threshold rules: red lines that mark the defined fairness_threshold in the chart."""
    stroke_color = (
        Threshold_Rule.stroke_accessible
        if accessibility_mode
        else Threshold_Rule.stroke
    )

    threshold_rule = (
        alt.Chart(threshold_df)
        .mark_rule(
            stroke=stroke_color,
            opacity=Threshold_Rule.opacity,
            strokeWidth=Threshold_Rule.stroke_width,
            tooltip="",
        )
        .encode(
            y=alt.value(Disparity_Chart.padding_y * chart_height),
            y2=alt.value((1 - Disparity_Chart.padding_y) * chart_height),
        )
    )

    lower_threshold_rule = threshold_rule.encode(
        x=alt.X("min:Q", scale=scales["x"]),
    )
    upper_threshold_rule = threshold_rule.encode(
        x=alt.X("max:Q", scale=scales["x"]),
    )

    return lower_threshold_rule + upper_threshold_rule


def __draw_threshold_bands(
    threshold_df,
    scales,
    chart_height,
    chart_width,
    accessibility_mode=False,
):
    """Draws threshold bands: regions painted red where the metric value is above the defined fairness_threshold."""
    fill_color = (
        Threshold_Band.color_accessible if accessibility_mode else Threshold_Band.color
    )

    threshold_band = (
        alt.Chart(threshold_df)
        .mark_rect(fill=fill_color, opacity=Threshold_Band.opacity, tooltip="")
        .encode(
            y=alt.value(Disparity_Chart.padding_y * chart_height),
            y2=alt.value((1 - Disparity_Chart.padding_y) * chart_height),
        )
    )

    lower_threshold_band = threshold_band.encode(
        x2="lower_end:Q",
        x=alt.X("min:Q", scale=scales["x"]),
    )
    upper_threshold_band = threshold_band.encode(
        x=alt.X("max:Q", scale=scales["x"]),
        x2="upper_end:Q",
    )

    return lower_threshold_band + upper_threshold_band


def __draw_threshold_text(
    fairness_threshold, ref_group, chart_height, accessibility_mode=False
):
    """Draw text that helps to read the threshold elements of the chart."""
    font_color = (
        Annotation.font_color if accessibility_mode else Annotation.font_color_threshold
    )

    threshold_text = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            baseline="top",
            align="left",
            font=FONT,
            fill=font_color,
            fontSize=Annotation.font_size,
            fontWeight=Annotation.font_weight,
            tooltip="",
        )
        .encode(
            x=alt.value(0),
            y=alt.value(chart_height * (1 - 2 / 3 * Disparity_Chart.padding_y)),
            text=alt.value(
                f"The metric value for any group should not be {fairness_threshold} (or more) times smaller or larger than that of the reference group {ref_group}."
            ),
        )
    )

    return threshold_text


def __get_threshold_elements(
    fairness_threshold,
    ref_group,
    scales,
    chart_height,
    chart_width,
    accessibility_mode=False,
):
    """Gets threshold elements (rules, bands and text) for the chart."""

    # CREATE THRESHOLD DF
    threshold_df = pd.DataFrame(
        {
            "min": -fairness_threshold + 1,
            "max": fairness_threshold - 1,
            "lower_end": scales["x"]["domain"][0],
            "upper_end": scales["x"]["domain"][1],
        },
        index=[0],
    )

    # RULES
    threshold_rules = __draw_threshold_rules(
        threshold_df, scales, chart_height, accessibility_mode
    )

    # BANDS
    threshold_bands = __draw_threshold_bands(
        threshold_df,
        scales,
        chart_height,
        chart_width,
        accessibility_mode,
    )

    # HELPER TEXT
    threshold_text = __draw_threshold_text(
        fairness_threshold, ref_group, chart_height, accessibility_mode
    )

    return threshold_rules + threshold_bands + threshold_text


def __draw_bubbles(
    plot_table,
    metrics,
    ref_group,
    scales,
    selection,
):
    """Draws the bubbles for all metrics."""

    # X AXIS GRIDLINES
    axis_values = __get_x_axis_values(scales["x"].domain, zero=False)

    x_axis = alt.Axis(
        values=axis_values, ticks=False, domain=False, labels=False, title=None
    )

    # COLOR
    bubble_color_encoding = alt.condition(
        selection,
        alt.Color("attribute_value:N", scale=scales["color"], legend=None),
        alt.value(Bubble.color_faded),
    )

    plot_table["tooltip_group_size"] = plot_table.apply(
        lambda row: get_tooltip_text_group_size(
            row["group_size"], row["total_entities"]
        ),
        axis=1,
    )

    # CHART INITIALIZATION
    bubble_centers = alt.Chart().mark_point()
    bubble_areas = alt.Chart().mark_circle()

    # LAYERING THE METRICS
    for metric in metrics:

        plot_table[f"tooltip_disparity_explanation_{metric}"] = plot_table.apply(
            lambda row: get_tooltip_text_disparity_explanation(
                row[f"{metric}_disparity_scaled"],
                row["attribute_value"],
                metric,
                ref_group,
            ),
            axis=1,
        )

        bubble_tooltip_encoding = [
            alt.Tooltip(field="attribute_value", type="nominal", title="Group"),
            alt.Tooltip(field="tooltip_group_size", type="nominal", title="Group Size"),
            alt.Tooltip(
                field=f"tooltip_disparity_explanation_{metric}",
                type="nominal",
                title="Disparity",
            ),
            alt.Tooltip(
                field=f"{metric}",
                type="quantitative",
                format=".2f",
                title=f"{metric}".upper(),
            ),
        ]

        # BUBBLE CENTERS
        trigger_centers = alt.selection_multi(empty="all", fields=["attribute_value"])

        bubble_centers += (
            alt.Chart(plot_table)
            .transform_calculate(metric_variable=f"'{metric.upper()}'")
            .mark_point(filled=True, size=Bubble.center_size)
            .encode(
                x=alt.X(f"{metric}_disparity_scaled:Q", scale=scales["x"], axis=x_axis),
                y=alt.Y("metric_variable:N", scale=scales["y"], axis=no_axis()),
                tooltip=bubble_tooltip_encoding,
                color=bubble_color_encoding,
                shape=alt.Shape(
                    "attribute_value:N", scale=scales["shape"], legend=None
                ),
            )
            .add_selection(trigger_centers)
        )

        # BUBBLE AREAS
        trigger_areas = alt.selection_multi(empty="all", fields=["attribute_value"])

        bubble_areas += (
            alt.Chart(plot_table)
            .mark_circle(opacity=Bubble.opacity)
            .transform_calculate(metric_variable=f"'{metric.upper()}'")
            .encode(
                x=alt.X(f"{metric}_disparity_scaled:Q", scale=scales["x"], axis=x_axis),
                y=alt.Y("metric_variable:N", scale=scales["y"], axis=no_axis()),
                tooltip=bubble_tooltip_encoding,
                color=bubble_color_encoding,
                size=alt.Size("group_size:Q", legend=None, scale=scales["bubble_size"]),
            )
            .add_selection(trigger_areas)
        )

    return bubble_areas + bubble_centers


def get_disparity_bubble_chart_components(
    plot_table,
    metrics,
    ref_group,
    global_scales,
    selection,
    fairness_threshold,
    chart_height,
    chart_width,
    accessibility_mode=False,
    concat_chart=False,
):
    """Creates the components necessary to plot the disparity chart."""

    # COMPUTE SCALED DISPARITIES
    for metric in metrics:
        plot_table[f"{metric}_disparity_scaled"] = plot_table.apply(
            lambda row: transform_ratio(row[f"{metric}_disparity"]), axis=1
        )

    # POSITION SCALES
    position_scales = __get_position_scales(
        plot_table, metrics, fairness_threshold, chart_height, chart_width
    )

    scales = dict(global_scales, **position_scales)

    # RULES
    horizontal_rules = __draw_metrics_rules(metrics, scales, concat_chart)
    reference_rule = __draw_reference_rule(ref_group, chart_height, chart_width)

    # LABELS
    x_ticks_labels = __draw_x_ticks_labels(scales, chart_height)

    # ANNOTATIONS
    text_annotations = __draw_text_annotations(
        ref_group, chart_height, scales["x"].range
    )

    # BUBBLES - CENTERS & AREAS
    bubbles = __draw_bubbles(
        plot_table,
        metrics,
        ref_group,
        scales,
        selection,
    )

    # THRESHOLD & BANDS
    if fairness_threshold is not None:
        threshold_elements = __get_threshold_elements(
            fairness_threshold,
            ref_group,
            scales,
            chart_height,
            chart_width,
            accessibility_mode,
        )

        # ASSEMBLE CHART WITH THRESHOLD
        main_chart = (
            horizontal_rules
            + reference_rule
            + x_ticks_labels
            + text_annotations
            + threshold_elements
            + bubbles
        )

    # ASSEMBLE CHART WITHOUT THRESHOLD
    else:
        main_chart = (
            horizontal_rules
            + x_ticks_labels
            + text_annotations
            + reference_rule
            + bubbles
        )

    return main_chart


def plot_disparity_bubble_chart(
    disparity_df,
    metrics_list,
    attribute,
    fairness_threshold=1.25,
    chart_height=None,
    chart_width=Disparity_Chart.full_width,
    accessibility_mode=False,
):
    """Draws bubble chart to visualize disparity in selected metrics versus that of a
    reference group for a given attribute.

    :param disparity_df: a dataframe generated by the Aequitas Bias class
    :type disparity_df: pandas.core.frame.DataFrame
    :param metrics_list: a list of the metrics of interest
    :type metrics_list: list
    :param attribute: an attribute to plot
    :type attribute: str
    :param fairness_threshold: a value for the maximum allowed disparity, defaults to 1.25
    :type fairness_threshold: float, optional
    :param chart_height: a value (in pixels) for the height of the chart
    :type chart_height: int, optional
    :param chart_width: a value (in pixels) for the width of the chart
    :type chart_width: int, optional
    :param accessibility_mode: a switch for the display of more accessible visual elements, defaults to False
    :type accessibility_mode: bool, optional

    :return: the full disparities chart
    :rtype: Altair chart object
    """

    (
        plot_table,
        metrics,
        ref_group,
        global_scales,
        chart_height,
        chart_width,
        selection,
    ) = Initializer.prepare_bubble_chart(
        disparity_df,
        metrics_list,
        attribute,
        fairness_threshold,
        chart_height,
        chart_width,
        Disparity_Chart,
        accessibility_mode,
    )

    # GET MAIN CHART COMPONENTS
    main_chart = get_disparity_bubble_chart_components(
        plot_table,
        metrics,
        ref_group,
        global_scales,
        selection,
        fairness_threshold,
        chart_height,
        chart_width,
        accessibility_mode,
    )

    # ADD LEGEND
    legend = draw_legend(global_scales, selection, chart_width)
    full_chart = main_chart + legend

    # FINALIZE CHART
    disparity_chart = (
        full_chart.configure_view(
            strokeWidth=0,
        )
        .configure_axisLeft(
            labelFontSize=Metric_Axis.label_font_size,
            labelColor=Metric_Axis.label_color,
            labelFont=FONT,
        )
        .configure_title(
            align="center",
            baseline="middle",
            font=FONT,
            fontWeight=Chart_Title.font_weight,
            fontSize=Chart_Title.font_size,
            color=Chart_Title.font_color,
        )
        .properties(
            height=chart_height,
            width=chart_width,
            title=f"Disparities on {attribute.title()}",
            padding=Disparity_Chart.full_chart_padding,
        )
        .resolve_scale(y="independent", size="independent")
    )

    return disparity_chart
