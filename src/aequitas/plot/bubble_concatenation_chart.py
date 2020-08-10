import math
import altair as alt
import pandas as pd

from aequitas.plot.bubble_disparity_chart import (
    get_disparity_bubble_chart_components,
)
from aequitas.plot.bubble_metric_chart import get_metric_bubble_chart_components

from aequitas.plot.commons.legend import draw_legend

from aequitas.plot.commons.style.classes import Title, Metric_Axis
from aequitas.plot.commons.style.text import FONT
import aequitas.plot.commons.style.sizes as Sizes
import aequitas.plot.commons.initializers as Initializer

# Default chart sizing options
CHART_PADDING = 0.1

# Altair 2.4.1 requires that all chart receive a dataframe, for charts that don't need it
# (like most annotations), we pass the following dummy dataframe to reduce the complexity of the resulting vega spec.
DUMMY_DF = pd.DataFrame({"a": [1, 1], "b": [0, 0]})


def __get_chart_sizes(chart_width):
    """ Calculates the widths of the disparity and metric charts that make-up the concatenated chart.
    The individual widths are calculated based on the provided desired overall chart width."""

    chart_sizes = dict(
        disparity_chart_width=0.6 * chart_width, metric_chart_width=0.4 * chart_width
    )
    return chart_sizes


def draw_chart_title(chart_title, chart_width):
    """Draws chart titles."""

    title = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="center",
            fill=Title.font_color,
            fontSize=Title.font_size,
            font=FONT,
            tooltip="",
        )
        .encode(
            x=alt.value(chart_width / 2),
            y=alt.value(Title.margin_top),
            text=alt.value(chart_title),
        )
    )

    return title


def plot_concatenated_bubble_charts(
    disparity_df,
    metrics_list,
    attribute,
    fairness_threshold=1.25,
    chart_height=None,
    chart_width=Sizes.Concat_Chart.full_width,
    chart_padding=CHART_PADDING,
    accessibility_mode=False,
):
    """Draws bubble chart to visualize the values of the selected metrics for a given attribute."""
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
        chart_padding,
        Sizes.Disparity_Chart,
        accessibility_mode,
    )

    chart_sizes = __get_chart_sizes(chart_width)

    # TITLES
    disparity_title = draw_chart_title(
        "DISPARITIES", chart_sizes["disparity_chart_width"]
    )
    metric_title = draw_chart_title("METRICS", chart_sizes["metric_chart_width"])

    # DISPARITY CHART
    disparity_chart = (
        (
            get_disparity_bubble_chart_components(
                plot_table,
                metrics,
                ref_group,
                global_scales,
                selection,
                fairness_threshold,
                chart_height,
                chart_sizes["disparity_chart_width"],
                chart_padding,
                accessibility_mode,
                concat_chart=True,
            )
            + disparity_title
        )
        .resolve_scale(y="independent", size="independent")
        .properties(height=chart_height, width=chart_sizes["disparity_chart_width"])
    )

    # METRIC CHART
    metric_chart = (
        (
            get_metric_bubble_chart_components(
                plot_table,
                metrics,
                ref_group,
                global_scales,
                selection,
                fairness_threshold,
                chart_height,
                chart_sizes["metric_chart_width"],
                chart_padding,
                accessibility_mode,
                concat_chart=True,
            )
            + metric_title
            + draw_legend(global_scales, selection, chart_sizes["metric_chart_width"])
        )
        .resolve_scale(y="independent")
        .properties(height=chart_height, width=chart_sizes["metric_chart_width"])
    )

    full_chart = (
        alt.hconcat(disparity_chart, metric_chart, bounds="flush", spacing=20)
        .configure_view(strokeWidth=0)
        .configure_axisLeft(
            labelFontSize=Metric_Axis.label_font_size,
            labelColor=Metric_Axis.label_color,
            labelFont=FONT,
        )
    )

    return full_chart
