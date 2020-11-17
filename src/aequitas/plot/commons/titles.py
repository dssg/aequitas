from aequitas.plot.commons.style.classes import Chart_Title
from aequitas.plot.commons.style.text import FONT


def get_title_configuration():
    return {
        "align": "center",
        "baseline": "middle",
        "font": FONT,
        "fontWeight": Chart_Title.font_weight,
        "fontSize": Chart_Title.font_size,
        "color": Chart_Title.font_color
    }
