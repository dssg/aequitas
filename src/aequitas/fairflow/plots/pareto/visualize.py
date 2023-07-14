"""Visualization module to display or save interactive visual application"""
import json
from datetime import datetime
from string import Template
from uuid import uuid4

import numpy as np
import pkg_resources


# NumPy data types are not JSON serializable. This custom JSON encoder will
# transform them to standard Python types.
class _CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return super().encode(bool(o))
        else:
            return super().default(o)


def _id_generator():
    return f"id-{str(uuid4())}"


def _generate_filename(tuner_type, alpha, performance_metric, fairness_metric):
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")

    return (
        "_".join(
            [tuner_type, str(alpha), performance_metric, fairness_metric, time_string]
        )
        + ".html"
    )


def _make_html(payload):
    div_id = _id_generator()
    template_path = pkg_resources.resource_filename(__name__, "template.html")
    pareto_js_path = pkg_resources.resource_filename(
        __name__, "js/dist/fairAutoML.js"
    )

    with open(pareto_js_path, "r") as file:
        pareto_js_bundle = file.read()

    with open(template_path, "r") as file:
        html_template = file.read()
        html_template = Template(html_template)

    return html_template.substitute(
        div_id=div_id,
        library_bundle=pareto_js_bundle,
        payload=json.dumps(payload, cls=_CustomJSONEncoder),
    )


def visualize(tuner, mode="display", save_path=None):
    """Render interactive application to explore results of hyperparameter optimization search.
    Tuner parameter must have a fairness metric set.

    Parameters
    ----------
    tuner : BaseTuner
        A tuner instance of one of the fairautoml.tuners.
    mode : str, optional, default "display"
       The mode can be "display" to render the visualization in a notebook, or "save" to export
       the visualization to an .html file.
    save_path : str, optional
        The save path for the exported .html of the application.

    """
    if tuner.fairness_metric is None:
        raise NameError(
            """No fairness metric was used in the hyperparameter optimization process.
            The visualization requires a fairness metric to be set."""
        )

    if "is_pareto" not in tuner.results.columns:
        tuner._compute_pareto_models()  # noqa

    try:
        tuner_results_flat = tuner.results.reset_index()
    except TypeError:
        print(
            """tuner.results is not a valid Pandas DataFrame.
            'visualize' must be run after the hyperparameter search has been completed."""
        )

    fairness_metrics = list(tuner.available_fairness_metrics)
    performance_metrics = list(
        tuner.available_performance_metrics - {"pp", "money_pp", "pn", "money_pn"}
    )

    filtered_results = tuner_results_flat[
        fairness_metrics
        + performance_metrics
        + ["model_id", "hyperparams", "is_pareto"]
    ]

    models_json = filtered_results.to_json(
        orient="records", force_ascii=False, double_precision=3
    )

    payload = {
        "models": models_json,
        "recommended_model": tuner.best_model_details,
        "optimized_fairness_metric": tuner.fairness_metric,
        "optimized_performance_metric": tuner.performance_metric,
        "fairness_metrics": fairness_metrics,
        "performance_metrics": performance_metrics,
        "tuner_type": tuner.__class__.__qualname__,
        "alpha": tuner.alpha,
    }

    app_html = _make_html(payload)

    if mode == "save":
        path = save_path or _generate_filename(
            tuner.__class__.__qualname__,
            tuner.alpha,
            tuner.performance_metric,
            tuner.fairness_metric,
        )

        with open(path, "w") as file:
            file.write(app_html)

        return "Visualization successfully exported to " + path + "."

    if mode == "display":
        try:
            from IPython.display import IFrame, display
        except ImportError as e:
            msg = """IPython is not available.
            The `visualize` in mode 'display' must be used in JupyterLab."""
            raise ImportError(msg) from e

        with open("pareto-viz.html", "w") as file:
            file.write(app_html)

        return display(IFrame(src="pareto-viz.html", width="100%", height=640))

    else:
        raise ValueError(
            'mode parameter must be "display" or "save" but received "' + mode + '".'
        )
