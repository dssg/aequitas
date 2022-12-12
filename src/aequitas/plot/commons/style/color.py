# COLORS
BLACK = "rgb(25, 39, 78)"
GRAY = "rgb(102, 111, 137)"
LIGHT_GRAY = "rgb(225, 226, 231)"
RED = "rgb(211, 47, 47)"
TEAL = "rgb(0, 163, 158)"

BLUE = "rgb(76, 136, 238)"
ORANGE = "rgb(254, 97, 0)"
PINK = "rgb(220, 38, 127)"
YELLOW = "rgb(255, 176, 0)"
PURPLE = "rgb(150, 95, 230)"

# NAMED COLORS
REFERENCE = GRAY
FADED = LIGHT_GRAY
THRESHOLD = RED
PARITY_PASS = TEAL
PARITY_FAIL = THRESHOLD
PARITY_RESULT_PALETTE = [PARITY_PASS, PARITY_FAIL]


# Colors will be selected for the groups in the chart by the
# order in this palette array. Therefore, the order should 
# maximize contrast for smaller subsets [1:n].
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
