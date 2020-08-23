from aequitas.plot.commons.helpers import format_number


def get_tooltip_text_group_size(group_size, total_entities):
    return f'{format_number(group_size)} ({"{0:.2f}".format(group_size / total_entities * 100)}%)'


def get_tooltip_text_disparity_explanation(
    disparity_value, attribute_value, metric, ref_group
):
    if attribute_value == ref_group:
        return "Reference group"

    if disparity_value > 0:
        return (
            "{0:.2f}".format(disparity_value + 1)
            + " times larger "
            + metric.upper()
            + " than the reference group"
        )
    elif disparity_value < 0:
        return (
            "{0:.2f}".format(-disparity_value + 1)
            + " times smaller "
            + metric.upper()
            + " than the reference group"
        )
    else:
        return f"Same {metric.upper()} as the reference group"


def get_tooltip_text_parity_test_explanation(parity_result, metric, fairness_threshold):
    if parity_result == "Reference":
        return "Reference group"
    if parity_result == "Pass":
        return f"Pass. Below fairness threshold of {fairness_threshold}"
    if parity_result == "Fail":
        return f"Fail. Above fairness threshold of {fairness_threshold}"
