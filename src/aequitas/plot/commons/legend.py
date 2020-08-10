import pandas as pd
import altair as alt
from aequitas.plot.commons.style.classes import Legend
from aequitas.plot.commons.style.text import FONT
from aequitas.plot.commons.helpers import no_axis


DUMMY_DF = pd.DataFrame({"a": [1, 1], "b": [0, 0]})


def draw_legend(global_scales, selection, chart_width):
    """Draws the interactive group's colors legend for the chart."""

    groups = global_scales["color"].domain
    labels = groups.copy()
    labels[0] = labels[0] + " [REF]"
    legend_df = pd.DataFrame({"attribute_value": groups, "label": labels})

    # Position the legend to the right of the chart
    x_position = chart_width * 1.1

    # Comfort text that explains how to interact with the legend.
    comfort_text = (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="left",
            baseline="middle",
            color=Legend.font_color,
            fontSize=Legend.font_size,
            font=FONT,
            fontWeight=Legend.font_weight,
            tooltip="",
        )
        .encode(
            x=alt.value(x_position),
            y=alt.value(0),
            text=alt.value("Click to highlight a group."),
        )
    )

    # Conditionally color each legend item
    # If the group is selected, it is colored according to the group color scale, otherwise it is faded
    color_encoding = alt.condition(
        selection,
        alt.Color("attribute_value:N", scale=global_scales["color"], legend=None),
        alt.value(Legend.color_faded),
    )

    # Offset the positioning of the legend items after the comfort text
    y_scale = alt.Scale(
        domain=groups,
        range=[
            Legend.font_size + Legend.vertical_spacing,  # comfort text size + spacing
            (Legend.font_size + Legend.vertical_spacing)  # comfort text size + spacing
            + (len(groups) * Legend.font_size)  # number of legend elements x text size
            + (
                (len(groups) + 1) * Legend.vertical_spacing
            ),  # (number of "spacings" + start and end "spacings") x spacing
        ],
    )

    # Draw color squares for each group
    circles = (
        alt.Chart(legend_df)
        .mark_point(filled=True, tooltip="")
        .encode(
            x=alt.value(x_position),
            y=alt.Y("attribute_value:N", scale=y_scale, axis=no_axis()),
            color=color_encoding,
            shape=alt.Shape(
                "attribute_value:N", scale=global_scales["shape"], legend=None
            ),
        )
        .add_selection(selection)
    )

    # Draw colored label for each group
    trigger_text = alt.selection_multi(empty="all", fields=["attribute_value"])

    text = (
        alt.Chart(legend_df)
        .mark_text(
            align="left",
            baseline="middle",
            font=FONT,
            fontSize=Legend.font_size,
            fontWeight=Legend.font_weight,
            tooltip="",
        )
        .encode(
            x=alt.value(x_position + Legend.vertical_spacing),
            y=alt.Y("attribute_value:N", scale=y_scale, axis=no_axis()),
            text=alt.Text("label:N"),
            color=color_encoding,
        )
        .add_selection(trigger_text)
    )

    return circles + text + comfort_text
