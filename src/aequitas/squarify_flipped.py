import matplotlib.pyplot as plt

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
    return max([max(rect['dx'] / rect['dy'], rect['dy'] / rect['dx']) for rect in layout(sizes, x, y, dx, dy)])


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
        print(f"while loop {i}")
    "while loop broken"
    current = sizes[:i]

    remaining = sizes[i:]

    (leftover_x, leftover_y, leftover_dx, leftover_dy) = leftover(current, x, y, dx, dy)
    print()
    print(f"about to layout {current} x:{x}, y:{y}, dx:{dx}, dy:{dy} -- {layout(current, x, y, dx, dy)}")
    print(
        f"squarifying the rest: {remaining}--leftover_x:{leftover_x}, leftover_y:{leftover_x}, leftover_dx:{leftover_dx}, leftover_dy:{leftover_dy}")
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
                        label=None, value=None, ax=None, **kwargs):
    """
    Plotting with Matplotlib from predefined rectangles. Adapted from squarify
    source code.

    :param rects: list-like of dictionaries indicating rectangle dimensions
        for plotting
    :param norm_x:  overall figure dimensions for normalizing value box sizes
    :param norm_y: overall figure dimensions for normalizing value box sizes
    :param color: color string or list-like of colors to use for value boxes
    :param label: list-like used as label text
    :param value: list-like used as value text
    :param ax: Matplotlib Axes instance
    :param kwargs: dict, keyword arguments passed to matplotlib.Axes.bar

    :return: matplotlib.Axis
    """
    x = [rect['x'] for rect in rects]
    y = [rect['y'] for rect in rects]
    dx = [rect['dx'] for rect in rects]
    dy = [rect['dy'] for rect in rects]

    ax.bar(x, dy, width=dx, bottom=y, color=color,
           label=label, align='edge', **kwargs)

    if value is not None:
        va = 'center' if label is None else 'top'
        for v, r in zip(value, rects):
            x, y, dx, dy = r['x'], r['y'], r['dx'], r['dy']
            ax.text(x + dx / 2, y + dy / 2, v, va=va, ha='center')

    if label is not None:
        va = 'center' if value is None else 'bottom'
        under_plot = []
        alphabet = list(map(chr, range(65, 91)))
        under_plot_num = 0

        for l, r in zip(label, rects):
            x, y, dx, dy = r['x'], r['y'], r['dx'], r['dy']
            length = dx

            indent_length = 4
            CHAR_PLACEHOLDER = 1.5

            # if box large enough, add labels and values
            if (dx >= (indent_length * 2) + CHAR_PLACEHOLDER * len(l)) & (dx > 10):
                ax.text(x + dx / 2, y + dy / 2, l, va=va, ha='center',
                        fontsize=14, wrap=False)

            else:
                # add labels that don't fit in boxes underneath plot
                ax.text(x + dx / 2, y + dy / 2, alphabet[under_plot_num], va=va,
                        ha='center', fontsize=10, wrap=False)
                underplot_label = l.replace('\n', ', ')
                under_plot.append(f"{alphabet[under_plot_num]}: {underplot_label}")
                under_plot_num += 1

    if len(under_plot) > 0:
        unlabeled = ('\n').join(under_plot)
        ax.text(0.0, -0.05, f"Not labeled above:\n{unlabeled}",
                ha='left', va='top', transform=ax.transAxes, fontsize=14)

    ax.set_xlim(0, norm_x)
    ax.set_ylim(0, norm_y)
    return ax