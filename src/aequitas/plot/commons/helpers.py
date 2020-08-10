import altair as alt
from millify import millify, prettify


def no_axis():
    """ Returns an invisible Altair axis. """
    return alt.Axis(domain=False, grid=False, ticks=False, labels=False, title=" ")


def format_number(value):
    """ Format number to be easily read by humans. """
    if value > 1000000:
        return millify(value, precision=2)
    return prettify(value)


def transform_ratio(value):
    """ Transformation that takes ratios and applies a function that preserves equal distances to origin (1) 
    for similar relationships, eg. a ratio of 2 (twice the size) is at the same distance of 1 (same size) as 
    0.5 (half the size).
    Read: 'how many times larger or smaller than reference disparity'."""

    if value >= 1:
        return value - 1
    else:
        return 1 - 1 / value


def calculate_chart_size_from_elements(header_size, element_size, number_elements):
    """ Calculates the total size for a chart based on the desired size for the header (title, axis, etc.), and the 
    size and number of elements (lines, columns). """

    return header_size + (element_size * number_elements)


def to_list(user_input):
    return user_input if (isinstance(user_input, list)) else [user_input]
