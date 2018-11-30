
'''
Attribution: Adapted from Uri Laserson's squaify source code for plotting treemaps in
matplotlib based on algorithm from Bruls, Huizing, van Wijk,
"Squarified Treemaps". (https://github.com/laserson/squarify)

Treemaps are flipped so smallest square is in bottom right corner. Added
function for plotting predefined rectangles. Made adjustments to treemap labeling.
'''

__author__ = "Pedro Saleiro <saleiro@uchicago.edu>, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."

import matplotlib.pyplot as plt
import itertools

def normalize_sizes(sizes, dx, dy):
    total_size = sum(sizes)
    total_area = dx * dy
    sizes = map(float, sizes)
    sizes = map(lambda size: size * total_area / total_size, sizes)
    return list(sizes)


def layoutrow(sizes, x, y, dx, dy):
    # generate rects for each size in sizes
    # dx >= dy
    # they will fill up height dy, and width will be determined by their area
    # sizes should be pre-normalized wrt dx * dy (i.e., they should be same units)
    covered_area = sum(sizes)
    width = covered_area / dy
    rects = []
    heights = 0
    for size in sizes:
        rects.append({'x': x, 'y': dy - (size / width), 'dx': width, 'dy': size / width})
        dy -= size / width
    return rects


def layoutcol(sizes, x, y, dx, dy):
    # generate rects for each size in sizes
    # dx < dy
    # they will fill up width dx, and height will be determined by their area
    # sizes should be pre-normalized wrt dx * dy (i.e., they should be same units)
    covered_area = sum(sizes)
    height = covered_area / dx
    rects = []
    for size in sizes:
        rects.append({'x': x, 'y': dy - height, 'dx': size / height, 'dy': height})
        x += size / height
    return rects


def layout(sizes, x, y, dx, dy):
    return layoutrow(sizes, x, y, dx, dy) if dx >= dy else layoutcol(sizes, x, y, dx, dy)


def leftoverrow(sizes, x, y, dx, dy):
    # compute remaining area when dx >= dy
    covered_area = sum(sizes)
    width = covered_area / dy
    leftover_x = x + width
    leftover_y = dy
    leftover_dx = dx - width
    leftover_dy = dy
    return (leftover_x, leftover_y, leftover_dx, leftover_dy)


def leftovercol(sizes, x, y, dx, dy):
    # compute remaining area when dx < dy
    covered_area = sum(sizes)
    height = covered_area / dx
    leftover_x = x
    leftover_y = dy
    leftover_dx = dx
    leftover_dy = dy - height
    return (leftover_x, leftover_y, leftover_dx, leftover_dy)


def leftover(sizes, x, y, dx, dy):
    return leftoverrow(sizes, x, y, dx, dy) if dx >= dy else leftovercol(sizes, x, y, dx, dy)


def worst_ratio(sizes, x, y, dx, dy):
    return max(
        max(
            rect['dx'] / rect['dy'],
            rect['dy'] / rect['dx']
            )
        for rect in layout(sizes, x, y, dx, dy)
    )


def squarify(sizes, x, y, dx, dy):
    # sizes should be pre-normalized wrt dx * dy (i.e., they should be same units)
    # or dx * dy == sum(sizes)
    # sizes should be sorted biggest to smallest
    sizes = list(map(float, sizes))

    if len(sizes) == 0:
        return []

    if len(sizes) == 1:
        return layout(sizes, x, y, dx, dy)

    # figure out where 'split' should be
    i = 1
    while i < len(sizes) and worst_ratio(sizes[:i], x, y, dx, dy) >= worst_ratio(sizes[:(i + 1)], x, y, dx, dy):
        i += 1
    current = sizes[:i]

    remaining = sizes[i:]

    (leftover_x, leftover_y, leftover_dx, leftover_dy) = leftover(current, x, y, dx, dy)

    return layout(current, x, y, dx, dy) + \
       squarify(remaining, leftover_x, leftover_y, leftover_dx, leftover_dy)


def pad_rectangle(rect):
    if rect['dx'] > 2:
        rect['x'] += 1
        rect['dx'] -= 2
    if rect['dy'] > 2:
        rect['y'] += 1
        rect['dy'] -= 2


def padded_squarify(sizes, x, y, dx, dy):
    rects = squarify(sizes, x, y, dx, dy)
    for rect in rects:
        pad_rectangle(rect)
    return rects

def squarify_plot_rects(rects, norm_x=100, norm_y=100, color=None,
                        labels=None, values=None, ax=None, **kwargs):
    """
    Plotting with Matplotlib from predefined rectangles. Adapted from squarify
    source code.

    :param rects: list-like of dictionaries indicating rectangle dimensions
        for plotting
    :param norm_x:  overall figure dimensions for normalizing value box sizes
    :param norm_y: overall figure dimensions for normalizing value box sizes
    :param color: color string or list-like of colors to use for value boxes
    :param labels: list-like used as label text
    :param values: list-like used as value text
    :param ax: Matplotlib Axes instance
    :param kwargs: dict, keyword arguments passed to matplotlib.Axes.bar

    :return: matplotlib.Axis
    """
    x = [rect['x'] for rect in rects]
    y = [rect['y'] for rect in rects]
    dx = [rect['dx'] for rect in rects]
    dy = [rect['dy'] for rect in rects]

    ax.bar(x, dy, width=dx, bottom=y, color=color,
           label=labels, align='edge', **kwargs)

    INDENT_LENGTH = 4
    CHAR_PLACEHOLDER = 1.5

    if values is not None:
        va = 'center' if labels is None else 'top'
        if values:
            plot_ready_values = [
                val if isinstance(val, str)
                else f"{val:.2}" if isinstance(val, (int, float))
                else ""
                for val in values
            ]

        for val, r in zip(plot_ready_values, rects):
            x, y, dx, dy = r['x'], r['y'], r['dx'], r['dy']

            # if box large enough, add labels and values
            if (dx >= (INDENT_LENGTH * 2) + (CHAR_PLACEHOLDER * len(val))) &\
                    (dx > 10):
                ax.text(x + dx / 2, y + dy / 2, val, va=va,
                        ha='center', fontsize=12)

    if labels is not None:
        va = 'center' if values is None else 'bottom'
        under_plot = []
        alphabet = list(map(chr, range(65, 91)))
        under_plot_num = 0

        if values:
            plot_ready_values = [
                val if isinstance(val, str)
                else f"{val:.2}" if isinstance(val, (int, float))
                else ""
                for val in values
            ]
        else:
            plot_ready_values = itertools.repeat(None)

        for (label, r, val) in zip(labels, rects, plot_ready_values):
            x, y, dx, dy = r['x'], r['y'], r['dx'], r['dy']
            length = dx

            # if box large enough, add labels and values
            if (dx >= (INDENT_LENGTH * 2) + CHAR_PLACEHOLDER * len(label)) & (dx > 10):
                ax.text(x + dx / 2, y + dy / 2, label, va=va, ha='center',
                        fontsize=14, wrap=False)

            else:
                # add labels that don't fit in boxes underneath plot
                ax.text(x + dx / 2, y + dy / 2, alphabet[under_plot_num], va=va,
                        ha='center', fontsize=12, wrap=False)

                underplot_label = str(label) if val is None else f"{label}, {val}"
                under_plot.append(f"{alphabet[under_plot_num]}: {underplot_label}")
                under_plot_num += 1


        if len(under_plot) > 0:
            unlabeled = ('\n').join(under_plot)
            ax.text(0.0, -0.05, f"Not labeled above:\n{unlabeled}",
                    ha='left', va='top', transform=ax.transAxes, fontsize=14)

    ax.set_xlim(0, norm_x)
    ax.set_ylim(0, norm_y)
    return ax
