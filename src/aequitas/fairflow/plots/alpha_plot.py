from matplotlib import pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("whitegrid", {"grid.linestyle": "--"})

# Create the plot for a specific dataset
plot_dataset = "baf_base"

# define the name of the metrics for plot
if "baf" in plot_dataset:
    perf_metric_plot = "TPR"
    fair_metric_plot = "Pred. Eq."
else:
    perf_metric_plot = "Acc."
    fair_metric_plot = "Dem. Par."

samples = bootstrap_alpha_results[plot_dataset]
alphas = np.linspace(0.0, 1, 101)
alpha_metric_mean = {}
ci_ub = {}
ci_lb = {}

for method in samples.keys():
    alpha_metric_mean[method] = {}
    ci_ub[method] = {}
    ci_lb[method] = {}
    for alpha in alphas:
        alpha_metric_mean[method][alpha] = np.mean(samples[method][alpha]["alpha_weighted"])
        ci_ub[method][alpha] = np.quantile(samples[method][alpha]["alpha_weighted"], 0.975)
        ci_lb[method][alpha] = np.quantile(samples[method][alpha]["alpha_weighted"], 0.025)

fig, axs = plt.subplots(1,1, figsize=(6.4*1, 4.8*1), dpi=200)
fig.tight_layout(pad=5.0)


for method in order:
    if method not in samples:
        continue
    line, = axs.plot(alpha_metric_mean[method].keys(), alpha_metric_mean[method].values(), label=methods_names[method])
    axs.fill_between(ci_lb[method].keys(), ci_lb[method].values(), ci_ub[method].values(), alpha=0.1)
axs.set_ylim([-0.05, 1.05])
axs.set_xlabel("Alpha")
axs.set_ylabel(f"α * {perf_metric_plot} + (1-α) * {fair_metric_plot}")
plt.text(-0.04, -0.19, f"({fair_metric_plot})", fontdict={"fontsize":5});
plt.text(0.98, -0.19, f"({perf_metric_plot})", fontdict={"fontsize":5});
plt.title(datasets_names[plot_dataset])

plt.legend();
plt.savefig(f"alpha_plot_{plot_dataset}.pdf", dpi=200, bbox_inches="tight")