from IPython.core.display import display, HTML
import pkg_resources
import numpy as np
import string
import json

from aequitas.plot.commons.helpers import to_list, transform_ratio

from aequitas.plot.commons.initializers import __filter_df, __sanitize_metrics


def __id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase)
    return "".join(np.random.choice(chars, size, replace=True))


def __make_plot_html(id, render_function_name, payload):
    aequitas_viz_lib_path = pkg_resources.resource_filename(
        __name__, "build/aequitas.js"
    )
    bundle = open(aequitas_viz_lib_path, "r", encoding="utf8").read()
    plot_html = """
	<head>
	</head>
	<body>
	    <script>
	    {bundle}
	    </script>
	    <div id="{id}">
	    </div>
	    <script>
	        aequitas.{render_function_name}("#{id}", {payload});
	    </script>
	</body>
	</html>
	""".format(
        bundle=bundle,
        id=id,
        payload=json.dumps(payload),
        render_function_name=render_function_name,
    )
    return plot_html


def __filter_df_alt(disparity_df, metrics, xy_plot=False):
    viz_fields = [
        "attribute_name",
        "attribute_value",
        "group_size",
        "total_entities",
    ]
    viz_fields += metrics

    if not xy_plot:
        viz_fields += [f"{metric}_disparity" for metric in metrics]
        viz_fields += [f"{metric}_ref_group_value" for metric in metrics]

    plot_table = disparity_df[viz_fields].copy(deep=True)

    return plot_table


def plot_disparity_bubble_chart(
    disparity_df,
    metrics_list,
    attribute=None,
    fairness_threshold=1.25,
    accessibility_mode=False,
    id=__id_generator(),
):

    metrics = __sanitize_metrics(metrics_list)

    plot_table = __filter_df_alt(disparity_df, metrics)

    for metric in metrics:
        plot_table[f"{metric}_disparity_scaled"] = plot_table.apply(
            lambda row: transform_ratio(row[f"{metric}_disparity"]), axis=1
        )

    payload = {
        "data": plot_table.to_dict("records"),
        "metrics": metrics,
        "attribute": attribute,
        "fairness_threshold": fairness_threshold,
        "accessibility_mode": accessibility_mode,
    }

    plot_html = __make_plot_html(id, "plotDisparityBubbleChart", payload)
    return display(HTML(plot_html))
