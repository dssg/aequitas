class Summary_Chart:
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


class Disparity_Chart:
    vertical_header_height = 100
    line_height = 150
    min_line_height = 80
    full_width = 800
    min_width = 400
    padding = 0.1

class Metric_Chart:
    vertical_header_height = 50
    line_height = 125
    min_line_height = 80
    full_width = 450
    min_width = 400
    padding = 0.1


class Concat_Chart:
    vertical_header_height = 125
    line_height = 100
    min_line_height = 80
    full_width = 950
    spacing = 20
    min_width = Metric_Chart.min_width + Disparity_Chart.min_width + spacing


class XY_Chart:
    full_side = 450
    min_side = 200
    padding = 0.05
