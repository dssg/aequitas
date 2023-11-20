import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from .plot import Plot

sns.set()
sns.set_style("whitegrid", {"grid.linestyle": "--"})


# Clean names for plotting
methods_names = {
    "lightgbm_baseline": "LGBM",
    "fairgbm_baf": "FGBM",
    "fairgbm_folktables": "FGBM",
    "exponentiated_gradient_baf": "EG",
    "exponentiated_gradient_folktables": "EG",
    "grid_search_baf": "GS",
    "grid_search_folktables": "GS",
    "group_threshold_baf": "GT",
    "group_threshold_folktables": "GT",
    "prevalence_oversampling": "OS",
    "prevalence_undersampling": "US",
}

datasets_names = {
    "baf_base": "Bank Account Fraud (Base)",
    "baf_variant_1": "Bank Account Fraud (Type I)",
    "baf_variant_2": "Bank Account Fraud (Type II)",
    "baf_variant_3": "Bank Account Fraud (Type III)",
    "baf_variant_4": "Bank Account Fraud (Type IV)",
    "baf_variant_5": "Bank Account Fraud (Type V)",
    "folktables_acsemployment": "ACS Employment",
    "folktables_acsincome": "ACS Income",
    "folktables_acsmobility": "ACS Mobility",
    "folktables_acspubliccoverage": "ACS Public Coverage",
    "folktables_acstraveltime": "ACS Travel Time",
}

# Order to plot methods
order = [
    "lightgbm_baseline",
    "fairgbm_baf",
    "fairgbm_folktables",
    "prevalence_oversampling",
    "prevalence_undersampling",
    "exponentiated_gradient_baf",
    "exponentiated_gradient_folktables",
    "group_threshold_baf",
    "group_threshold_folktables",
    "grid_search_baf",
    "grid_search_folktables",
]


def visualize(plot: Plot):
    # define the name of the metrics for plot
    if "baf" in plot.dataset:
        perf_metric_plot = "TPR"
        fair_metric_plot = "Pred. Eq."
    else:
        perf_metric_plot = "Acc."
        fair_metric_plot = "Dem. Par."

    alphas = np.linspace(0.0, 1, 101)
    alpha_metric_mean = {}
    ci_ub = {}
    ci_lb = {}

    for method in plot.bootstrap_results.keys():
        alpha_metric_mean[method] = {}
        ci_ub[method] = {}
        ci_lb[method] = {}
        for alpha in alphas:
            alpha_metric_mean[method][alpha] = np.mean(
                plot.bootstrap_results[method][alpha]["alpha_weighted"]
            )
            ci_ub[method][alpha] = np.quantile(
                plot.bootstrap_results[method][alpha]["alpha_weighted"], 0.975
            )
            ci_lb[method][alpha] = np.quantile(
                plot.bootstrap_results[method][alpha]["alpha_weighted"], 0.025
            )

    fig, axs = plt.subplots(1, 1, figsize=(6.4 * 1, 4.8 * 1), dpi=200)
    fig.tight_layout(pad=5.0)

    for method in order:
        if method not in plot.bootstrap_results:
            continue
        (line,) = axs.plot(
            alpha_metric_mean[method].keys(),
            alpha_metric_mean[method].values(),
            label=methods_names[method],
        )
        axs.fill_between(
            ci_lb[method].keys(),
            ci_lb[method].values(),
            ci_ub[method].values(),
            alpha=0.1,
        )
    axs.set_ylim([-0.05, 1.05])
    axs.set_xlabel("Alpha")
    axs.set_ylabel(f"α * {perf_metric_plot} + (1-α) * {fair_metric_plot}")
    plt.text(-0.04, -0.19, f"({fair_metric_plot})", fontdict={"fontsize": 5})
    plt.text(0.98, -0.19, f"({perf_metric_plot})", fontdict={"fontsize": 5})
    plt.title(datasets_names[plot.dataset])

    plt.legend()
