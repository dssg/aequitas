# COLORS
GRAY = "rgb(117,117,117)"
LIGHT_GRAY = "rgb(224,224,224)"
RED = "rgb(217, 46, 28)"
TEAL = "rgb(20, 152, 181)"

BLUE = "rgb(100, 143, 255)"
ORANGE = "rgb(254, 97, 0)"
PINK = "rgb(220, 38, 127)"
YELLOW = "rgb(255, 176, 0)"
PURPLE = "rgb(120, 94, 240)"

# NAMED COLORS
REFERENCE = GRAY
FADED = LIGHT_GRAY
THRESHOLD = RED
PARITY_PASS = TEAL
PARITY_FAIL = THRESHOLD
PARITY_RESULT_PALETTE = [PARITY_PASS, PARITY_FAIL]


# Colors will be selected for the groups in the chart by the
# order that they are in this palette array. Therefore, the
# order should maximamize constrast for smaller subsets [1:n].
CATEGORICAL_PALETTE_COLOR_SAFE = [
    BLUE,
    ORANGE,
    PINK,
    YELLOW,
    PURPLE,
]

CATEGORICAL_PALETTE_ALTERNATIVE = [
    "rgb(106, 172, 250)",
    "rgb(62, 204, 156)",
    "rgb(255, 163, 31)",
    "rgb(255, 109, 135)",
    "rgb(182, 135, 231)",
]

CATEGORICAL_COLOR_PALETTE = CATEGORICAL_PALETTE_COLOR_SAFE
