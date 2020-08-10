from zaiquitas.plot.commons.helpers import format_number


def get_tooltip_text_group_size(plot_table):
    population_size = plot_table["total_entities"].iloc[0]

    def __format_value_to_string(value):
        return f'{format_number(value)} ({"{0:.2f}".format(value / population_size * 100)}%)'

    group_size_tooltip_strings = plot_table.apply(
        lambda row: __format_value_to_string(row["group_size"]), axis=1
    )

    return group_size_tooltip_strings


def get_tooltip_text_disparity_explanation(plot_table, metric, ref_group):
    def __get_value_explanation(row, metric):
        value = row[f"{metric}_disparity_scaled"]

        if row["attribute_value"] == ref_group:
            return "Reference group"

        if value > 0:
            return (
                "{0:.2f}".format(value + 1)
                + " times larger "
                + metric.upper()
                + " than the reference group"
            )
        elif value < 0:
            return (
                "{0:.2f}".format(-value + 1)
                + " times smaller "
                + metric.upper()
                + " than the reference group"
            )
        else:
            return f"Same {metric.upper()} as the reference group"

    disparity_explanations = plot_table.apply(
        lambda row: __get_value_explanation(row, metric), axis=1,
    )
    return disparity_explanations
