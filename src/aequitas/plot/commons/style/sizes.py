class Chart:
    full_chart_padding = {"top": 15, "bottom": 10}
    padding_x = 0.05
    padding_y = 0.1


class Summary_Chart(Chart):
    vertical_header_height = 45
    line_height = 85
    min_line_height = 50
    max_line_height = 150
    line_height_ratio = 0.65
    horizontal_header_width = 80
    column_width = 240
    min_column_width = 150
    max_column_width = 300
    column_width_ratio = 0.75


class Disparity_Chart(Chart):
    vertical_header_height = 100
    line_height = 150
    min_line_height = 80
    full_width = 800
    min_width = 380


class Metric_Chart(Chart):
    vertical_header_height = 50
    line_height = 125
    min_line_height = 80
    full_width = 800
    min_width = 380


class Concat_Chart(Chart):
    vertical_header_height = 110
    line_height = 75
    min_line_height = 50
    full_width = 760
    spacing = 0
    min_width = Metric_Chart.min_width + Disparity_Chart.min_width + spacing


class XY_Chart(Chart):
    full_side = 450
    min_side = 200
    padding_y = 0.05
