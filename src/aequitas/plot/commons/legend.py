import pandas as pd
import altair as alt
import math

from aequitas.plot.commons.style.classes import Legend
from aequitas.plot.commons.style.text import FONT
from aequitas.plot.commons.helpers import no_axis
from aequitas.plot.commons import labels as Label


DUMMY_DF = pd.DataFrame({"a": [1, 1], "b": [0, 0]})


def draw_legend(global_scales, selection, chart_width):
    """Draws the interactive group's colors legend for the chart."""

    groups = global_scales["color"].domain
    labels = groups.copy()
    labels[0] = f"{labels[0]} {Label.REF_INDICATOR}"
    legend_df = pd.DataFrame({"attribute_value": groups, "label": labels})

    # Position the legend to the right of the chart

    title_text_x_position = chart_width
    title_text_height = Legend.title_font_size + Legend.row_padding

    radius = math.sqrt(Legend.symbol_size / math.pi)
    entries_circles_x_position = title_text_x_position + radius + Legend.offset
    entries_text_x_position = title_text_x_position + radius * 2 + Legend.offset * 3 - 1

    # Title of the legend.
    title_text = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="left",
            baseline=Legend.title_baseline,
            color=Legend.title_font_color,
            fontSize=Legend.title_font_size,
            font=FONT,
            fontWeight=Legend.title_font_weight,
        )
        .encode(
            x=alt.value(title_text_x_position),
            y=alt.value(Legend.margin),
            text=alt.value(Label.MULTIPLE_GROUPS),
        )
    )

    # Subtitle text that explains how to interact with the legend.
    subtitle_text = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="left",
            baseline=Legend.title_baseline,
            color=Legend.title_font_color,
            fontSize=Legend.title_font_size,
            font=FONT,
            fontWeight=Legend.title_font_weight,
        )
        .encode(
            x=alt.value(title_text_x_position),
            y=alt.value(Legend.margin + title_text_height),
            text=alt.value(Label.CLICK_HIGHLIGHT),
        )
    )

    # Conditionally color each legend item
    # If the group is selected, it is colored according to the group color scale, otherwise it is faded
    color_encoding = alt.condition(
        selection,
        alt.Color("attribute_value:N", scale=global_scales["color"], legend=None),
        alt.value(Legend.color_faded),
    )

    # Offset the positioning of the legend items after the subtitle text
    legend_start_y_position = Legend.margin + title_text_height + Legend.font_size + Legend.title_padding

    y_scale = alt.Scale(
        domain=groups,
        range=[
            legend_start_y_position,
            legend_start_y_position
            # number of legend elements x text size
            + (len(groups) * Legend.font_size)
            # number of legend elements x spacing
            + (len(groups) * Legend.offset),
        ],
    )

    # Draw colored symbols for each group
    entries_circles = (
        alt.Chart(legend_df)
        .mark_point(filled=True, opacity=1, size=Legend.symbol_size, cursor=Legend.cursor)
        .encode(
            x=alt.value(entries_circles_x_position),
            y=alt.Y("attribute_value:N", scale=y_scale, axis=no_axis()),
            color=color_encoding,
            shape=alt.Shape(
                "attribute_value:N", scale=global_scales["shape"], legend=None
            ),
        )
        .add_selection(selection)
    )
    trigger_text = alt.selection_multi(empty="all", fields=["attribute_value"])

    # Draw colored label for each group
    entries_text = (
        alt.Chart(legend_df)
        .mark_text(
            align="left",
            baseline="middle",
            font=FONT,
            fontSize=Legend.font_size,
            fontWeight=Legend.font_weight,
            cursor=Legend.cursor,
        )
        .encode(
            x=alt.value(entries_text_x_position),
            y=alt.Y("attribute_value:N", scale=y_scale, axis=no_axis()),
            text=alt.Text("label:N"),
            color=color_encoding,
        )
        .add_selection(trigger_text)
    )

    return entries_circles + entries_text + subtitle_text + title_text
