import altair as alt
import numpy as np

from millify import millify, prettify


def no_axis():
    """Returns an invisible Altair axis."""
    return alt.Axis(domain=False, grid=False, ticks=False, labels=False, title=None)


def format_number(value):
    """Format number to be easily read by humans."""
    if value > 1000000:
        return millify(value, precision=2)
    return prettify(value)


def transform_ratio(value):
    """Transformation that takes ratios and applies a function that preserves equal distances to origin (1)
    for similar relationships, eg. a ratio of 2 (twice the size) is at the same distance of 1 (same size) as
    0.5 (half the size).
    Read: 'how many times larger or smaller than reference disparity'."""
    if value >= 1:
        return value - 1
    elif value == 0:
        return np.nan
    else:
        return 1 - 1 / value


def calculate_chart_size_from_elements(header_size, element_size, number_elements):
    """Calculates the total size for a chart based on the desired size for the header (title, axis, etc.), and the
    size and number of elements (lines, columns)."""
    return header_size + (element_size * number_elements)


def to_list(user_input):
    """Wrap a particular `user_input` in a list."""
    return user_input if (isinstance(user_input, list)) else [user_input]


def get_chart_metadata(filename):
    """Generate the metadata for a single Altair chart."""
    return {
        # Vega-Embed (https://github.com/vega/vega-embed) options:
        "embedOptions": {
            # The multiplier for the width and height
            # to export a higher resolution image.
            "scaleFactor": 5,
            # The filename (without extension) for the
            # chart when exported in PNG or SVG format.
            "downloadFileName": filename,
        }
    }
