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

metrics_names = {
    "Predictive Equality": "Pred. Eq.",
    "Equal Opportunity": "Eq. Opp.",
    "Demographic Parity": "Dem. Par.",
    "TPR": "TPR",
    "FPR": "FPR",
    "FNR": "FNR",
    "Accuracy": "Acc.",
    "Precision": "Prec.",
}


def visualize(plot: Plot):
    # define the name of the metrics for plot
    perf_metric_plot = metrics_names[plot.performance_metric]
    fair_metric_plot = metrics_names[plot.fairness_metric]

    x = plot.x

    x_metric_mean = {}
    ci_ub = {}
    ci_lb = {}

    ub = 1 - ((1 - plot.confidence_intervals) / 2)
    lb = (1 - plot.confidence_intervals) / 2
    for method in plot.bootstrap_results.keys():
        x_metric_mean[method] = {}
        ci_ub[method] = {}
        ci_lb[method] = {}
        for x_point in x:
            x_metric_mean[method][x_point] = np.mean(
                plot.bootstrap_results[method][x_point]["alpha_weighted"]
            )
            ci_ub[method][x_point] = np.quantile(
                plot.bootstrap_results[method][x_point]["alpha_weighted"], ub
            )
            ci_lb[method][x_point] = np.quantile(
                plot.bootstrap_results[method][x_point]["alpha_weighted"], lb
            )

    fig, axs = plt.subplots(1, 1, figsize=(6.4 * 1, 4.8 * 1), dpi=200)
    fig.tight_layout(pad=5.0)

    for method in order:
        if method not in plot.bootstrap_results:
            continue
        (line,) = axs.plot(
            x_metric_mean[method].keys(),
            x_metric_mean[method].values(),
            label=methods_names[method],
        )
        axs.fill_between(
            ci_lb[method].keys(),
            ci_lb[method].values(),
            ci_ub[method].values(),
            alpha=0.1,
        )
    axs.set_ylim([-0.05, 1.05])
    if plot.plot_type == "alpha":
        axs.set_xlabel("Alpha")
        axs.set_ylabel(f"α * {perf_metric_plot} + (1-α) * {fair_metric_plot}")
        plt.text(-0.04, -0.19, f"({fair_metric_plot})", fontdict={"fontsize": 5})
        plt.text(0.98, -0.19, f"({perf_metric_plot})", fontdict={"fontsize": 5})
    else:
        axs.set_xlabel("Bootstrap Size")
        if plot.kwargs["alpha_points"] == 0.0:
            axs.set_ylabel(f"{fair_metric_plot}")
        elif plot.kwargs["alpha_points"] == 1.0:
            axs.set_ylabel(f"{perf_metric_plot}")
        else:
            axs.set_ylabel(f"α * {perf_metric_plot} + (1-α) * {fair_metric_plot}")
    plt.title(datasets_names[plot.dataset])

    plt.legend()
