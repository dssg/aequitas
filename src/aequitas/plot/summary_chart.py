import math
import altair as alt
import pandas as pd

from aequitas.plot.commons.helpers import (
    no_axis,
    transform_ratio,
    calculate_chart_size_from_elements,
    to_list,
    format_number,
)
from aequitas.plot.commons.tooltips import (
    get_tooltip_text_group_size,
    get_tooltip_text_disparity_explanation,
    get_tooltip_text_parity_test_explanation,
)
from aequitas.plot.commons.style.classes import (
    Title,
    Subtitle,
    Parity_Result,
    Annotation,
    Legend,
)
from aequitas.plot.commons.style.text import FONT
from aequitas.plot.commons.style.sizes import Summary_Chart
from aequitas.plot.commons import initializers as Initializer
from aequitas.plot.commons import validators as Validator

# Altair 2.4.1 requires that all chart receive a dataframe, for charts that don't need it
# (like most annotations), we pass the following dummy dataframe to reduce the complexity of the resulting vega spec.
DUMMY_DF = pd.DataFrame({"a": [1, 1], "b": [0, 0]})


def __get_scales(max_num_groups):
    """Creates an Altair scale for the color of the parity test result, and another
    for the x axis of the group circles subplot."""

    scales = dict()

    # COLOR
    scales["color"] = alt.Scale(
        domain=["Reference", "Pass", "Fail"], range=Parity_Result.color_palette
    )

    # GROUP CIRCLES X AXIS
    scales["circles_x"] = alt.Scale(
        domain=[-max_num_groups, max_num_groups], nice=False
    )

    return scales


def __get_size_constants(
    chart_height, chart_width, num_attributes, num_metrics, max_num_groups
):
    """Calculates the heights, widths and spacings of the components of the summary chart
    based on the provided desired overall chart height and width, as well as the number of
    attributes (columns) and metrics (lines)."""

    size_constants = dict(
        # Chart sizes
        attribute_titles_height=0.15 * chart_height,
        line_spacing=0.2 * chart_height / num_metrics,
        line_height=Summary_Chart.line_height_ratio * chart_height / num_metrics,
        metric_titles_width=0.1 * chart_width,
        column_spacing=0.15 * chart_width / num_attributes,
        column_width=Summary_Chart.column_width_ratio * chart_width / num_attributes,
        # Circle size
        ## Conditional definition of the size where for each additional unit in
        ## max_num_groups, we subtract 25 squared pixels from the area of the
        ## circle, which has the base value of 350 for 0 groups. From max_num_groups
        ## equal to 13 or more, we keep the size at the minimum value of 25 to
        ## make sure the circles are visible.
        group_circle_size=-25 * max_num_groups + 350 if max_num_groups < 13 else 25,
    )
    return size_constants


def __draw_attribute_title(attribute, width, size_constants):
    """Draws a single attribute's title."""

    return (
        alt.Chart(DUMMY_DF)
        .mark_text(
            align="center",
            baseline="middle",
            font=FONT,
            size=Title.font_size,
            color=Title.font_color,
            fontWeight=Title.font_weight,
        )
        .encode(
            text=alt.value(attribute.title()),
        )
        .properties(width=width, height=size_constants["attribute_titles_height"])
    )


def __draw_metric_line_titles(metrics, size_constants):
    """Draws left hand side titles for metrics."""

    metric_line_titles = []

    for metric in metrics:
        # METRIC TITLE
        metric_title = (
            alt.Chart(DUMMY_DF)
            .transform_calculate(y_position="1.2")
            .mark_text(
                align="center",
                baseline="middle",
                font=FONT,
                fontWeight=Title.font_weight,
                size=Title.font_size,
                color=Title.font_color,
            )
            .encode(
                alt.Y("y_position:Q", scale=alt.Scale(domain=[3, 1]), axis=no_axis()),
                text=alt.value(metric.upper()),
            )
        )

        # GROUPS TEXT
        group_circles_title = (
            alt.Chart(DUMMY_DF)
            .transform_calculate(y_position="2")
            .mark_text(
                align="center",
                baseline="middle",
                font=FONT,
                size=Subtitle.font_size,
                color=Subtitle.font_color,
            )
            .encode(
                alt.Y("y_position:Q", scale=alt.Scale(domain=[3, 1]), axis=no_axis()),
                text=alt.value("Groups"),
            )
        )

        # PERCENT. POP TEXT
        population_percentage_title = (
            alt.Chart(DUMMY_DF)
            .transform_calculate(y_position="2.7")
            .mark_text(
                align="center",
                baseline="middle",
                font=FONT,
                size=Subtitle.font_size,
                color=Subtitle.font_color,
            )
            .encode(
                alt.Y("y_position:Q", scale=alt.Scale(domain=[3, 1]), axis=no_axis()),
                text=alt.value("% Pop."),
            )
        )

        metric_line_titles.append(
            (
                metric_title + group_circles_title + population_percentage_title
            ).properties(
                height=size_constants["line_height"],
                width=size_constants["metric_titles_width"],
            )
        )

    # EMPTY CORNER SPACE
    # To make sure that the attribute columns align properly with the title column, we need to create a blank
    # space of the same size of the attribute titles. For this purpose, we use the same function (__draw_attribute_title)
    # and pass in an empty string so that nothing is actually drawn.
    top_left_corner_space = __draw_attribute_title(
        "", size_constants["metric_titles_width"], size_constants
    )

    # CONCATENATE SUBPLOTS
    metric_titles = alt.vconcat(
        top_left_corner_space,
        *metric_line_titles,
        spacing=size_constants["line_spacing"],
        bounds="flush",
    )

    return metric_titles


def __get_parity_result_variable(row, metric, fairness_threshold):
    """ Creates parity test result variable for each provided row, separating the Reference group from the passing ones."""
    if row["attribute_value"] == row["ref_group_value"]:
        return "Reference"
    elif abs(row[f"{metric}_disparity_scaled"]) < fairness_threshold - 1:
        return "Pass"
    else:
        return "Fail"


def __draw_parity_result_text(parity_result, color_scale):
    """Draws the uppercased text result of the provided parity test (Pass, Fail or Reference),
    color-coded according to the provided Altair scale."""

    return (
        alt.Chart(pd.DataFrame({"parity_result": parity_result}, index=[0]))
        .transform_calculate(y_position="1")
        .mark_text(
            align="center",
            baseline="middle",
            font=FONT,
            size=Parity_Result.font_size,
            fontWeight=Parity_Result.font_weight,
        )
        .encode(
            alt.Y("y_position:Q", scale=alt.Scale(domain=[3, 1]), axis=no_axis()),
            alt.Color(
                "parity_result:O", scale=color_scale, legend=alt.Legend(title="")
            ),
            text=alt.value(parity_result.upper()),
        )
    )


def __draw_population_bar(population_bar_df, metric, color_scale):
    """ Draws a stacked bar of the sum of the percentage of population of the groups that obtained each result for the parity test."""
    population_bar_tooltips = [
        alt.Tooltip(field=f"{metric}_parity_result", type="nominal", title="Parity"),
        alt.Tooltip(
            field="tooltip_group_size",
            type="nominal",
            title="Size",
        ),
        alt.Tooltip(field="tooltip_groups_name_size", type="nominal", title="Groups"),
    ]

    population_bar = (
        alt.Chart(population_bar_df)
        .transform_calculate(y_position="2.8")
        .mark_bar(size=6, stroke="white")
        .encode(
            alt.X("sum(group_size):Q", stack="normalize", axis=no_axis()),
            alt.Y("y_position:Q", scale=alt.Scale(domain=[3, 1]), axis=no_axis()),
            alt.Color(
                f"{metric}_parity_result:O",
                scale=color_scale,
                legend=alt.Legend(
                    title="Parity Test",
                    padding=20,
                ),
            ),
            tooltip=population_bar_tooltips,
        )
    )

    return population_bar


def __draw_group_circles(plot_df, metric, scales, size_constants):
    """Draws a circle for each group, color-coded by the result of the parity test.
    The groups are spread around the central reference group according to their disparity."""

    circle_tooltip_encoding = [
        alt.Tooltip(field="attribute_value", type="nominal", title="Group"),
        alt.Tooltip(field="tooltip_group_size", type="nominal", title="Group Size"),
        alt.Tooltip(
            field=f"tooltip_parity_test_explanation_{metric}",
            type="nominal",
            title="Parity Test",
        ),
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

    return (
        alt.Chart(plot_df)
        .transform_calculate(y_position="2")
        .mark_circle(opacity=1)
        .encode(
            alt.X(
                f"{metric}_disparity_rank:Q", scale=scales["circles_x"], axis=no_axis()
            ),
            alt.Y("y_position:Q", scale=alt.Scale(domain=[3, 1]), axis=no_axis()),
            alt.Color(
                f"{metric}_parity_result:O",
                scale=scales["color"],
                legend=alt.Legend(title=""),
            ),
            size=alt.value(size_constants["group_circle_size"]),
            tooltip=circle_tooltip_encoding,
        )
    )


def __draw_parity_test_explanation(fairness_threshold, x_position):
    """Draw text that explains what does pass/fail mean in the parity test results."""

    explanation_text = alt.Chart(DUMMY_DF).mark_text(
        baseline="top",
        align="left",
        font=FONT,
        fill=Annotation.font_color,
        fontSize=Annotation.font_size,
        fontWeight=Annotation.font_weight,
    )

    explanation_text_group = explanation_text.encode(
        x=alt.value(x_position),
        y=alt.value(0),
        text=alt.value(
            f"For a group to pass the parity test its disparity to the reference group cannot exceed the fairness threshold ({fairness_threshold})."
        ),
    )

    explanation_text_attribute = explanation_text.encode(
        x=alt.value(x_position),
        y=alt.value(Annotation.font_size * Annotation.line_spacing),
        text=alt.value(
            f"An attribute passes the parity test for a given metric if all its groups pass the test."
        ),
    )

    return explanation_text_group + explanation_text_attribute


def __create_population_bar_df(attribute_df, metric):
    """Creates a pandas aggregation of the attribute_df by parity result, along with the
    list of groups tooltip variable."""

    attribute_df["group_size_formatted"] = attribute_df.apply(
        lambda row: format_number(row["group_size"]), axis=1
    )
    attribute_df["tooltip_groups_name_size"] = (
        attribute_df["attribute_value"]
        + " ("
        + attribute_df["group_size_formatted"].astype(str)
        + ")"
    )
    population_bar_df = (
        attribute_df.groupby(by=f"{metric}_parity_result")
        .agg(
            {
                "attribute_name": min,
                "total_entities": min,
                "group_size": sum,
                "tooltip_groups_name_size": lambda x: ", ".join(x),
            }
        )
        .reset_index()
    )

    population_bar_df["tooltip_group_size"] = population_bar_df.apply(
        lambda row: get_tooltip_text_group_size(
            row["group_size"], row["total_entities"]
        ),
        axis=1,
    )

    return population_bar_df


def __create_group_rank_variable(attribute_df, metric):
    """ Creates the disparity rank variable for the given metric, centered around 0 (the Reference Group's value). """

    # RANK
    attribute_df[f"{metric}_disparity_rank"] = attribute_df[
        f"{metric}_disparity_scaled"
    ].rank(method="first")

    # REFERENCE GROUP RANK
    reference_rank = attribute_df.loc[
        attribute_df[f"{metric}_parity_result"] == "Reference"
    ][f"{metric}_disparity_rank"].iloc[0]

    # CENTERED RANK
    attribute_df[f"{metric}_disparity_rank"] = (
        attribute_df[f"{metric}_disparity_rank"] - reference_rank
    )


def __create_tooltip_variables(attribute_df, metric, fairness_threshold):
    """ Creates disparity explanation and formatted group size tooltip variables. """

    # PARITY TEST EXPLANATION
    attribute_df[f"tooltip_parity_test_explanation_{metric}"] = attribute_df.apply(
        lambda row: get_tooltip_text_parity_test_explanation(
            row[f"{metric}_parity_result"],
            metric,
            fairness_threshold,
        ),
        axis=1,
    )

    # DISPARITY EXPLANATION
    ref_group = attribute_df["ref_group_value"].iloc[0]

    attribute_df[f"tooltip_disparity_explanation_{metric}"] = attribute_df.apply(
        lambda row: get_tooltip_text_disparity_explanation(
            row[f"{metric}_disparity_scaled"],
            row["attribute_value"],
            metric,
            ref_group,
        ),
        axis=1,
    )
    # FORMATTED GROUP SIZE

    attribute_df["tooltip_group_size"] = attribute_df.apply(
        lambda row: get_tooltip_text_group_size(
            row["group_size"], row["total_entities"]
        ),
        axis=1,
    )


def __create_disparity_variables(attribute_df, metric, fairness_threshold):
    """ Creates scaled disparity, parity test result & disparity explanation tooltip variables. """

    # SCALED DISPARITY VALUE
    attribute_df[f"{metric}_disparity_scaled"] = attribute_df.apply(
        lambda row: transform_ratio(row[f"{metric}_disparity"]), axis=1
    )

    # PARITY RESULT
    attribute_df[f"{metric}_parity_result"] = attribute_df.apply(
        __get_parity_result_variable,
        metric=metric,
        fairness_threshold=fairness_threshold,
        axis=1,
    )


def __get_attribute_column(
    attribute_df, metrics, scales, attribute, size_constants, fairness_threshold
):
    """ Returns a vertical concatenation of all elements of all metrics for each attribute's column."""

    metric_summary = []

    for metric in metrics:
        # CREATE VARIABLES IN DF
        __create_disparity_variables(attribute_df, metric, fairness_threshold)
        __create_tooltip_variables(attribute_df, metric, fairness_threshold)
        __create_group_rank_variable(attribute_df, metric)

        # PARITY RESULT TEXT
        ## The parity result is equal to the "worst" of each group's results
        ## If one group fails the parity test, the whole metric fails (for that attribute)
        parity_result = attribute_df.loc[
            attribute_df[f"{metric}_parity_result"] != "Reference"
        ][f"{metric}_parity_result"].min()

        parity_result_text = __draw_parity_result_text(parity_result, scales["color"])

        # GROUP CIRCLES
        group_circles = __draw_group_circles(
            attribute_df, metric, scales, size_constants
        )

        # POPULATION BAR
        population_bar_df = __create_population_bar_df(attribute_df, metric)
        population_bar = __draw_population_bar(
            population_bar_df, metric, scales["color"]
        )

        # LAYERING
        metric_summary.append(
            (parity_result_text + population_bar + group_circles)
            .properties(
                width=size_constants["column_width"],
                height=size_constants["line_height"],
            )
            .resolve_scale(x="independent")
        )

    # ATTRIBUTE TITLE
    attribute_title = __draw_attribute_title(
        attribute, size_constants["column_width"], size_constants
    )

    return alt.vconcat(
        attribute_title,
        *metric_summary,
        bounds="flush",
        spacing=size_constants["line_spacing"],
    )


def plot_summary_chart(
    disparity_df,
    metrics_list,
    attributes_list=None,
    fairness_threshold=1.25,
    chart_height=None,
    chart_width=None,
):
    """Draws chart that summarizes the parity results for the provided metrics across the existing attributes.
    This includes an overall result, the specific results by each attribute's groups as well as the percentage
    of population by result.

    :param disparity_df: a dataframe generated by the Aequitas Bias class
    :type disparity_df: pandas.core.frame.DataFrame
    :param metrics_list: a list of the metrics of interest
    :type metrics_list: list
    :param attributes_list: a list of the attributes of interest, defaults to using all in the dataframe
    :type attributes_list: list, optional
    :param fairness_threshold: a value for the maximum allowed disparity, defaults to 1.25
    :type fairness_threshold: float, optional
    :param chart_height: a value (in pixels) for the height of the chart
    :type chart_height: int, optional
    :param chart_width: a value (in pixels) for the width of the chart
    :type chart_width: int, optional

    :return: the full summary chart
    :rtype: Altair chart object
    """

    ## If a specific list of attributes was not passed, use all from df
    (
        metrics,
        attributes,
        chart_height,
        chart_width,
    ) = Initializer.prepare_summary_chart(
        disparity_df,
        metrics_list,
        attributes_list,
        fairness_threshold,
        chart_height,
        chart_width,
        Summary_Chart,
    )

    num_metrics = len(metrics)
    num_attributes = len(attributes)

    max_num_groups = max(
        disparity_df.loc[disparity_df["attribute_name"].isin(attributes)]
        .groupby(by="attribute_name")["attribute_value"]
        .count()
    )

    size_constants = __get_size_constants(
        chart_height, chart_width, num_attributes, num_metrics, max_num_groups
    )

    Validator.chart_size_summary(size_constants, num_metrics, num_attributes)

    # SCALES
    scales = __get_scales(max_num_groups)

    # METRIC TITLES
    metric_titles = __draw_metric_line_titles(metrics, size_constants)

    # RELEVANT FIELDS
    viz_fields = [
        "attribute_name",
        "attribute_value",
        "group_size",
        "total_entities",
        f"{metrics[0]}_ref_group_value",
        *metrics,
    ]
    viz_fields += [f"{metric}_disparity" for metric in metrics]

    attribute_columns = []

    for attribute in attributes:
        # CREATE ATTRIBUTE DF
        attribute_df = (
            disparity_df[viz_fields]
            .loc[disparity_df["attribute_name"] == attribute]
            .copy(deep=True)
        )

        attribute_df.rename(
            columns={f"{metrics[0]}_ref_group_value": "ref_group_value"}, inplace=True
        )

        # ATTRIBUTE COLUMN
        attribute_column = __get_attribute_column(
            attribute_df,
            metrics,
            scales,
            attribute,
            size_constants,
            fairness_threshold,
        )

        attribute_columns.append((attribute_column))

    # CONCATENATE ATTRIBUTE COLUMNS
    summary_chart_columns = alt.hconcat(
        *attribute_columns,
        bounds="flush",
        spacing=size_constants["column_spacing"] + size_constants["column_width"],
    )
    # ADD METRIC TITLES
    summary_chart_table = alt.hconcat(
        metric_titles,
        summary_chart_columns,
        bounds="flush",
        spacing=size_constants["metric_titles_width"]
        + size_constants["column_spacing"],
    )

    summary_chart_explanation = __draw_parity_test_explanation(
        fairness_threshold, size_constants["column_spacing"] / 2
    )

    full_summary_chart = (
        alt.vconcat(summary_chart_table, summary_chart_explanation)
        .properties(padding=Summary_Chart.full_chart_padding)
        .configure_legend(
            labelFont=FONT,
            labelColor=Legend.font_color,
            labelFontSize=Legend.font_size,
            titleFont=FONT,
            titleColor=Legend.font_color,
            titleFontSize=Legend.title_font_size,
            titleFontWeight=Legend.title_font_weight,
            titlePadding=Legend.title_margin_bottom + Legend.vertical_spacing,
        )
        .configure_view(strokeWidth=0)
    )

    return full_summary_chart
