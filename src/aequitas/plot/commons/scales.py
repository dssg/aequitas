from aequitas.plot.commons.style.classes import Shape, Bubble
from aequitas.plot.commons.style.sizes import Disparity_Chart
from aequitas.plot.commons.style import color as Colors
import altair as alt

import math


def get_chart_size_range(length, padding):
    """Calculate the chart size range for a given axis based on the length and padding."""

    return [length * padding, length * (1 - padding)]


def get_color_scale(plot_table, ref_group):
    """
    Calculate the color scale for the bubbles. If there are more groups than the length
    of the categorical color palette, then we use the same color for all the non-reference
    groups. Otherwise, we use an encoding where each group is given its own distinct color.
    The reference group bubble is always given the reference color.
    """
    num_groups = plot_table["attribute_value"].nunique()
    num_colors = len(Colors.CATEGORICAL_COLOR_PALETTE)

    if num_colors >= num_groups - 1:
        color_range = [Colors.REFERENCE] + Colors.CATEGORICAL_COLOR_PALETTE
    else:
        # Non-reference groups are blue. We repeat the color blue per each different group
        color_range = [Colors.REFERENCE] + [
            Colors.BLUE for group in range(1, num_groups)
        ]

    color_domain = list(plot_table["attribute_value"])
    color_domain.insert(0, color_domain.pop(color_domain.index(ref_group)))
    color_scale = alt.Scale(domain=color_domain, range=color_range)
    return color_scale


def get_shape_scale(plot_table, ref_group, accessibility_mode=False):
    """
    Calculate the shape scale for the bubbles centers. The reference group bubble center
    is a cross while the others are circles. If accessibility_mode is on and there are
    more colors in the categorical color palette than different non-reference groups , then we
    also use shapes to distinguish between colors in the scale.
    """

    num_groups = plot_table["attribute_value"].nunique()
    num_colors = len(Colors.CATEGORICAL_COLOR_PALETTE)

    if (num_colors >= num_groups - 1) and accessibility_mode:
        # we find the midpoint in the palette, first half will use the default shape
        # while the second half will use the alternative shape. By the order of the palette,
        # first half should have high constrat between colors, and second half will have
        # some colors that may be similar to the ones in the first half. Therefore, we
        # we change shape to make them more distinguishable in accessibility mode.
        palette_midpoint = math.ceil(num_colors / 2.0)
        shape_range = (
            [Shape.reference]
            + [Shape.default] * palette_midpoint
            + [Shape.alternative] * (num_colors - palette_midpoint)
        )
    else:
        shape_range = [Shape.reference] + [Shape.default] * (num_groups - 1)

    shape_domain = list(plot_table["attribute_value"])
    shape_domain.insert(0, shape_domain.pop(shape_domain.index(ref_group)))
    shape_scale = alt.Scale(domain=shape_domain, range=shape_range)
    return shape_scale


def get_bubble_size_scale(plot_table, metrics, chart_height):
    """Create the scale for the bubble size"""

    max_bubble_radius = (chart_height * Bubble.max_bubble_ratio) / len(metrics)
    bubble_size_domain = [0, plot_table["group_size"].max() * 1.2]
    bubble_size_range = [0, max_bubble_radius ** 2 * math.pi]

    bubble_size_scale = alt.Scale(domain=bubble_size_domain, range=bubble_size_range)
    return bubble_size_scale
