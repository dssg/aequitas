from aequitas.plot.commons.style import color as Color
from aequitas.plot.commons.style import text as Text


class Bubble:
    opacity = 0.2
    color_faded = Color.FADED
    center_size = 60
    # From bubble size scale: (1 - 2 * chart_padding) / 4
    max_bubble_ratio = 0.2


class Title:
    font_size = Text.FONT_SIZE_REGULAR
    font_weight = Text.FONT_WEIGHT_BOLD
    font_color = Color.GRAY
    margin_top = -15


class Chart_Title(Title):
    font_size = Text.FONT_SIZE_EXTRA_LARGE
    font_weight = Text.FONT_WEIGHT_BOLD
    margin_top = -10


class Subtitle:
    font_size = Text.FONT_SIZE_SMALL
    font_color = Color.GRAY


class Annotation:
    font_size = Text.FONT_SIZE_SMALL
    font_weight = Text.FONT_WEIGHT_REGULAR
    font_color = Color.GRAY
    font_color_reference = Color.REFERENCE
    font_color_threshold = Color.THRESHOLD
    line_spacing = 1.25


class Rule:
    stroke_width = 1.25
    stroke = Color.GRAY


class Reference_Rule(Rule):
    stroke_dash = [5, 5]
    stroke_width = 1
    stroke = Color.REFERENCE


class Threshold_Rule(Rule):
    stroke = Color.THRESHOLD
    stroke_accessible = Color.GRAY
    opacity = 0.8


class Axis(Rule):
    title_color = Color.GRAY
    title_font_size = Text.FONT_SIZE_LARGE
    title_font_weight = Text.FONT_WEIGHT_REGULAR
    label_color = Color.GRAY
    label_font_size = Text.FONT_SIZE_SMALL
    label_font_weight = Text.FONT_WEIGHT_REGULAR


class Metric_Axis(Axis):
    offset = 20
    label_font_size = Text.FONT_SIZE_LARGE
    label_padding = -30
    label_padding_concat_chart = -10
    label_angle = 0


class Scatter_Axis(Axis):
    title_padding = 30
    title_font_size = Text.FONT_SIZE_LARGE
    title_font_weight = Text.FONT_WEIGHT_REGULAR


class Threshold_Band:
    color = Color.THRESHOLD
    color_accessible = Color.GRAY
    opacity = 0.1


class Legend:
    margin_top = 30
    color_faded = Color.FADED
    font_color = Color.GRAY
    font_size = Text.FONT_SIZE_SMALL
    font_weight = Text.FONT_WEIGHT_REGULAR
    vertical_spacing = 10
    horizontal_spacing = 4
    title_font_size = Text.FONT_SIZE_REGULAR
    title_font_weight = Text.FONT_WEIGHT_REGULAR
    title_margin_bottom = 5
    circle_radius = 4


class Parity_Result:
    font_size = Text.FONT_SIZE_EXTRA_LARGE
    font_weight = Text.FONT_WEIGHT_BOLD
    color_palette = [Color.REFERENCE] + Color.PARITY_RESULT_PALETTE


class Shape:
    reference = "cross"
    default = "circle"
    alternative = "square"
