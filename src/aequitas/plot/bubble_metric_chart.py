import altair as alt
import pandas as pd

from aequitas.plot.commons.helpers import (
    no_axis,
    transform_ratio,
    get_chart_metadata,
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
    Threshold_Band,
    Threshold_Rule,
    Annotation,
    Bubble,
    Rule,
    Chart_Title,
)
from aequitas.plot.commons.style.sizes import Metric_Chart
from aequitas.plot.commons.style.text import FONT, FONT_SIZE_SMALL
from aequitas.plot.commons import initializers as Initializer
from aequitas.plot.commons import labels as Label

# Altair 2.4.1 requires that all chart receive a dataframe, for charts that don't need it
# (like most annotations), we pass the following dummy dataframe to reduce the complexity of the resulting vega spec.
DUMMY_DF = pd.DataFrame({"a": [1, 1], "b": [0, 0]})


def __get_position_scales(metrics, chart_height, chart_width, concat_chart):
    """Computes the scales for x and y encodings to be used in the metric bubble chart."""

    position_scales = dict()

    # METRICS VALUES SCALE
    x_range = get_chart_size_range(chart_width, Metric_Chart.padding_x)
    x_domain = [0, 1]
    position_scales["x"] = alt.Scale(domain=x_domain, range=x_range)

    # METRICS LABELS SCALE
    y_range = get_chart_size_range(chart_height, Metric_Chart.padding_y)
    y_domain = [metric.upper() for metric in metrics]
    position_scales["y"] = alt.Scale(domain=y_domain, range=y_range)

    return position_scales


def __draw_metrics_rules(metrics, scales, concat_chart):
    """Draws an horizontal rule for each metric where the bubbles will be positioned."""

    metrics_labels = [metric.upper() for metric in metrics]

    if concat_chart:
        y_axis = no_axis()
    else:
        y_axis = alt.Axis(
            domain=False,
            ticks=False,
            orient="left",
            labelAngle=Metric_Axis.label_angle,
            labelPadding=Metric_Axis.label_padding,
            title=None,
        )

    horizontal_rules = (
        alt.Chart(pd.DataFrame({"y_position": metrics_labels, "start": 0, "end": 1}))
        .mark_rule(
            strokeWidth=Metric_Axis.stroke_width,
            stroke=Metric_Axis.stroke,
            tooltip=None,
        )
        .encode(
            y=alt.Y("y_position:N", scale=scales["y"], axis=y_axis),
            x=alt.X("start:Q", scale=scales["x"]),
            x2="end:Q",
        )
    )

    return horizontal_rules


def __draw_domain_rules(scales):
    """Draws vertical lines at the ends of each metric's domain."""

    x_domain = [0, 1]
    domain_rules_df = pd.DataFrame({"x_position": x_domain})

    vertical_domain_rules = (
        alt.Chart(domain_rules_df)
        .mark_rule(
            strokeWidth=Rule.stroke_width,
            stroke=Rule.stroke,
            tooltip=None,
        )
        .encode(
            x=alt.X("x_position:Q", scale=scales["x"]),
            y=alt.value(scales["y"]["range"][0]),
            y2=alt.value(scales["y"]["range"][1]),
        )
    )

    return vertical_domain_rules


def __draw_x_ticks_labels(scales, chart_height):
    """Draws the numbers in the horizontal axis."""

    axis_values = [0, 0.25, 0.5, 0.75, 1]

    axis_df = pd.DataFrame({"value": axis_values})

    x_ticks_labels = (
        alt.Chart(axis_df)
        .mark_text(
            tooltip=None,
            align="center",
            fontSize=Axis.label_font_size,
            font=FONT,
            fontWeight=Axis.label_font_weight,
            color=Axis.label_color,
        )
        .encode(
            text=alt.Text("value:N"),
            x=alt.X(
                "value:Q",
                scale=scales["x"],
            ),
            y=alt.value(Metric_Chart.padding_y * chart_height * 0.7),
        )
    )

    return x_ticks_labels


def __draw_threshold_rules(threshold_df, scales, position, accessibility_mode=False):
    """Draws fairness threshold rules: red lines that mark the defined fairness threshold in the chart."""
    stroke_color = (
        Threshold_Rule.stroke_accessible
        if accessibility_mode
        else Threshold_Rule.stroke
    )

    threshold_rules = (
        alt.Chart(threshold_df)
        .mark_rect(
            stroke=stroke_color,
            opacity=Threshold_Rule.opacity,
            strokeWidth=Threshold_Rule.stroke_width,
            tooltip=None,
        )
        .encode(
            x=alt.X(
                field=f"{position}_threshold_value",
                type="quantitative",
                scale=scales["x"],
            ),
            x2=f"{position}_threshold_value:Q",
            y=alt.Y(field="metric", type="nominal", scale=scales["y"], axis=no_axis()),
        )
    )
    return threshold_rules


def __draw_threshold_bands(threshold_df, scales, accessibility_mode=False):
    """Draws fairness threshold bands: regions painted red where the metric value is above the defined fairness threshold."""

    fill_color = (
        Threshold_Band.color_accessible if accessibility_mode else Threshold_Band.color
    )

    lower_threshold_band = (
        alt.Chart(threshold_df)
        .mark_rect(fill=fill_color, opacity=Threshold_Band.opacity, tooltip=None)
        .encode(
            y=alt.Y(field="metric", type="nominal", scale=scales["y"], axis=no_axis()),
            x=alt.X("lower_threshold_value:Q", scale=scales["x"]),
            x2="lower_end:Q",
        )
    )

    upper_threshold_band = (
        alt.Chart(threshold_df)
        .mark_rect(fill=fill_color, opacity=Threshold_Band.opacity, tooltip=None)
        .encode(
            y=alt.Y(field="metric", type="nominal", scale=scales["y"], axis=no_axis()),
            x=alt.X("upper_threshold_value:Q", scale=scales["x"]),
            x2="upper_end:Q"
            # scales["x"]["range"][1]
            # Replicating the approach of the lower_threshold_band doesn't work...
        )
    )

    return lower_threshold_band + upper_threshold_band


def __draw_threshold_text(
    fairness_threshold, ref_group, chart_height, accessibility_mode=False, warnings=()
):
    """Draw text that helps to read the fairness threshold elements of the chart."""
    font_color = (
        Annotation.font_color if accessibility_mode else Annotation.font_color_threshold
    )
    warn_text = alt.Chart(DUMMY_DF).mark_text(
        baseline="top",
        align="left",
        font=FONT,
        fill=font_color,
        fontSize=Annotation.font_size,
        fontWeight=Annotation.font_weight,
    )
    n_warnings = 0
    text_explanation = []
    for group, metric in warnings:
        y_size = chart_height * (1 - 2 / 3 * Metric_Chart.padding_y) + Annotation.font_size * Annotation.line_spacing * (n_warnings + 1)
        explanation_text_warning = warn_text.encode(
            x=alt.value(0),
            y=alt.value(y_size),
            text=alt.value(
                f"Groups {group} have {metric} of 0 (zero). This "
                "does not allow for the calculation of relative disparities. "
                "The tooltip for these groups in this plot does not have relative disparities.",
            )
        )
        n_warnings +=1
        text_explanation.append(explanation_text_warning)
    threshold_text = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            baseline="top",
            align="left",
            font=FONT,
            fill=font_color,
            fontSize=Annotation.font_size,
            fontWeight=Annotation.font_weight,
            tooltip=None,
        )
        .encode(
            x=alt.value(0),
            y=alt.value(chart_height * (1 - 2 / 3 * Metric_Chart.padding_y)),
            text=alt.value(
                f"The metric value for any group should not be {fairness_threshold} (or more) times smaller or larger than that of the reference group {ref_group}."
            ),
        )
    )

    text_explanation.append(threshold_text)

    return alt.layer(*(chart for chart in text_explanation))


def __get_threshold_elements(
    plot_table,
    metrics,
    fairness_threshold,
    ref_group,
    scales,
    chart_height,
    concat_chart,
    accessibility_mode=False,
    warnings=(),
):
    """Gets fairness threshold elements (rules, bands and text) for the chart."""

    lower_values = []
    upper_values = []

    # CREATE THRESHOLD DF
    for metric in metrics:
        # Get reference value for each metric
        ref_group_value = plot_table.loc[plot_table["attribute_value"] == ref_group][
            metric
        ].iloc[0]

        # Enforce max upper boundary on 1
        upper_values.append(min(ref_group_value * fairness_threshold, 1))

        lower_values.append(ref_group_value / fairness_threshold)

    # Convert to uppercase to match bubbles' Y axis
    metrics_labels = [metric.upper() for metric in metrics]

    threshold_df = pd.DataFrame(
        {
            "lower_end": 0,
            "upper_end": 1,
            "lower_threshold_value": lower_values,
            "upper_threshold_value": upper_values,
            "metric": metrics_labels,
        }
    )

    # RULES
    upper_threshold_rules = __draw_threshold_rules(
        threshold_df, scales, "upper", accessibility_mode
    )
    lower_threshold_rules = __draw_threshold_rules(
        threshold_df, scales, "lower", accessibility_mode
    )

    # BANDS
    threshold_bands = __draw_threshold_bands(threshold_df, scales, accessibility_mode)

    # HELPER TEXT
    if concat_chart:
        return lower_threshold_rules + upper_threshold_rules + threshold_bands
    else:
        threshold_text = __draw_threshold_text(
            fairness_threshold, ref_group, chart_height, accessibility_mode, warnings
        )

        return (
            lower_threshold_rules
            + upper_threshold_rules
            + threshold_bands
            + threshold_text
        )


def __draw_bubbles(
    plot_table,
    metrics,
    ref_group,
    scales,
    selection,
):
    """Draws the bubbles for all metrics."""

    # X AXIS GRIDLINES
    axis_values = [0.25, 0.5, 0.75]
    x_axis = alt.Axis(
        values=axis_values, ticks=False, domain=False, labels=False, title=None, gridColor=Axis.grid_color
    )

    # COLOR
    bubble_color_encoding = alt.condition(
        selection,
        alt.Color("attribute_value:N", scale=scales["color"], legend=None),
        alt.value(Bubble.color_faded),
    )

    # CHART INITIALIZATION
    bubble_centers = alt.Chart().mark_point()
    bubble_areas = alt.Chart().mark_circle()

    plot_table["tooltip_group_size"] = plot_table.apply(
        lambda row: get_tooltip_text_group_size(
            row["group_size"], row["total_entities"]
        ),
        axis=1,
    )
    # LAYERING THE METRICS
    for metric in metrics:
        # TOOLTIP
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
            alt.Tooltip(field="attribute_value", type="nominal", title=Label.SINGLE_GROUP),
            alt.Tooltip(field="tooltip_group_size", type="nominal", title=Label.GROUP_SIZE),
            alt.Tooltip(
                field=f"tooltip_disparity_explanation_{metric}",
                type="nominal",
                title=Label.DISPARITY,
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
            .mark_point(filled=True, size=Bubble.center_size, cursor=Bubble.cursor)
            .encode(
                x=alt.X(f"{metric}:Q", scale=scales["x"], axis=x_axis),
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
            .mark_circle(opacity=Bubble.opacity, cursor=Bubble.cursor)
            .transform_calculate(metric_variable=f"'{metric.upper()}'")
            .encode(
                x=alt.X(f"{metric}:Q", scale=scales["x"], axis=x_axis),
                y=alt.Y("metric_variable:N", scale=scales["y"], axis=no_axis()),
                tooltip=bubble_tooltip_encoding,
                color=bubble_color_encoding,
                size=alt.Size("group_size:Q", legend=None, scale=scales["bubble_size"]),
            )
            .add_selection(trigger_areas)
        )

    return bubble_areas + bubble_centers


def get_metric_bubble_chart_components(
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
    """Creates the components necessary to plot the metric chart."""

    # Check for warnings in plot
    metric_warnings = []
    # COMPUTE SCALED DISPARITIES
    for metric in metrics:
        zero_metric_groups = plot_table[plot_table[f"{metric}_disparity"] == 0]
        zero_values = zero_metric_groups["attribute_value"].values
        if zero_values.any():
            metric_warnings.append([zero_values, metric])

        plot_table[f"{metric}_disparity_scaled"] = plot_table.apply(
            lambda row: transform_ratio(row[f"{metric}_disparity"]), axis=1
        )

    # POSITION SCALES
    position_scales = __get_position_scales(
        metrics, chart_height, chart_width, concat_chart
    )
    scales = dict(global_scales, **position_scales)

    # RULES
    horizontal_rules = __draw_metrics_rules(metrics, scales, concat_chart)
    vertical_domain_rules = __draw_domain_rules(scales)

    # LABELS
    x_ticks_labels = __draw_x_ticks_labels(scales, chart_height)

    # BUBBLES - CENTERS & AREAS
    bubbles = __draw_bubbles(
        plot_table,
        metrics,
        ref_group,
        scales,
        selection,
    )

    # THRESHOLD, BANDS & ANNOTATION
    if fairness_threshold is not None:
        threshold_elements = __get_threshold_elements(
            plot_table,
            metrics,
            fairness_threshold,
            ref_group,
            scales,
            chart_height,
            concat_chart,
            accessibility_mode,
            metric_warnings
        )
        # ASSEMBLE CHART WITH THRESHOLD
        main_chart = (
            horizontal_rules
            + vertical_domain_rules
            + x_ticks_labels
            + threshold_elements
            + bubbles
        )
    # ASSEMBLE CHART WITHOUT THRESHOLD
    else:
        main_chart = horizontal_rules + vertical_domain_rules + x_ticks_labels + bubbles

    return main_chart


def plot_metric_bubble_chart(
    disparity_df,
    metrics_list,
    attribute,
    fairness_threshold=1.25,
    chart_height=None,
    chart_width=Metric_Chart.full_width,
    accessibility_mode=False,
):
    """Draws bubble chart to visualize the values of the selected metrics for a given attribute.

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

    :return: the full metrics chart
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
        Metric_Chart,
        accessibility_mode,
    )
    # GET MAIN CHART COMPONENTS
    main_chart = get_metric_bubble_chart_components(
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
    metric_chart = (
        full_chart.configure_view(strokeWidth=0)
        .configure_axisLeft(
            labelFontSize=Metric_Axis.label_font_size,
            labelColor=Metric_Axis.label_color,
            labelFont=FONT,
        )
        .properties(
            height=chart_height,
            width=chart_width,
            title=f"Absolute values by {attribute.title()}",
            # padding=Metric_Chart.full_chart_padding,
            padding={
                "top": Metric_Chart.full_chart_padding,
                "bottom": -FONT_SIZE_SMALL * 1.25/3 * len(metrics_list) + Metric_Chart.full_chart_padding,
                "left": Metric_Chart.full_chart_padding,
                "right": Metric_Chart.full_chart_padding,
            },
            usermeta=get_chart_metadata("absolute_chart"),
        )
        .configure_title(
            font=FONT,
            fontWeight=Chart_Title.font_weight,
            fontSize=Chart_Title.font_size,
            color=Chart_Title.font_color,
            anchor=Chart_Title.anchor,
        )
        .resolve_scale(y="independent", size="independent")
    )

    return metric_chart
