import logging
import math
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm

from aequitas import squarify_flipped as sf

logging.getLogger(__name__)

__author__ = "Pedro Saleiro <saleiro@uchicago.edu>, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."



# module-level function
def assemble_ref_groups(disparities_table, ref_group_flag='_ref_group_value',
                         specific_measures=None, label_score_ref=None):
    """
    Creates a dictionary of reference groups for each metric in a data_table.

   :param disparities_table: a disparity table. Output of bias.get_disparity or
        fairness.get_fairness functions
    :param ref_group_flag: string indicating column indicates reference group
        flag value. Default is '_ref_group_value'.
    :param specific_measures: Limits reference dictionary to only specified
        metrics in a data table. Default is None.
    :param label_score_ref: Defines a metric, ex: 'fpr' (false positive rate)
        from which to mimic reference group for label_value and score. Used for
        statistical significance calculations in Bias() class. Default is None.

    :return: A dictionary
    """
    ref_groups = {}
    ref_group_cols = \
        set(disparities_table.columns[disparities_table.columns.str.contains(
            ref_group_flag)])

    # Note: specific measures is a set
    if specific_measures:
        if len(specific_measures) < 1:
            raise ValueError("At least one metric must be passed for which to "
                             "find refrence group.")

        specific_measures = specific_measures.union({label_score_ref})
        ref_group_cols = {measure + ref_group_flag for measure in specific_measures if
             measure + ref_group_flag in ref_group_cols}

    attributes = list(disparities_table.attribute_name.unique())

    for attribute in attributes:
        attr_table = \
            disparities_table.loc[disparities_table['attribute_name'] == attribute]

        attr_refs = {}
        for col in ref_group_cols:
            if col in ('label' + ref_group_flag, 'score' + ref_group_flag):
                continue

            metric_key = col.replace(ref_group_flag, '')
            attr_refs[metric_key] = \
                attr_table.loc[attr_table['attribute_name'] == attribute, col].min()
        if label_score_ref:
            is_valid_label_ref = lambda label: label + ref_group_flag in disparities_table.columns

            if not is_valid_label_ref(label_score_ref):
                try:
                    label_score_ref = next(measure for measure in specific_measures if is_valid_label_ref(measure))
                    logging.warning("The specified reference measure for label "
                                    "value and score is not included in the "
                                    f"data frame. Using '{label_score_ref}' "
                                    "reference group as label value and score "
                                    "reference instead.")

                except StopIteration:
                    raise ValueError("None of metrics passed in 'specific_measures' are in dataframe.")

            attr_refs['label_value'] = attr_refs[label_score_ref]
            attr_refs['score'] = attr_refs[label_score_ref]

        ref_groups[attribute] = attr_refs

    return ref_groups


# Plot() class
class Plot(object):
    """
    Plotting object allows for visualization of absolute group bias metrics and
    relative disparities calculated by Aequitas Group(), Bias(), and Fairness()
    class instances.
    """
    default_absolute_metrics = ('pprev', 'ppr', 'fdr', 'for', 'fpr', 'fnr')
    default_disparities = ('pprev_disparity', 'ppr_disparity',
                           'fdr_disparity', 'for_disparity',
                           'fpr_disparity', 'fnr_disparity')

    # Define mapping for conditional coloring based on fairness
    # determinations
    _metric_parity_mapping = {
        'ppr_disparity': 'Statistical Parity',
        'pprev_disparity': 'Impact Parity',
        'precision_disparity': 'Precision Parity',
        'fdr_disparity': 'FDR Parity',
        'for_disparity': 'FOR Parity',
        'fpr_disparity': 'FPR Parity',
        'fnr_disparity': 'FNR Parity',
        'tpr_disparity': 'TPR Parity',
        'tnr_disparity': 'TNR Parity',
        'npv_disparity': 'NPV Parity',
        'ppr': 'Statistical Parity',
        'pprev': 'Impact Parity',
        'precision': 'Precision Parity',
        'fdr': 'FDR Parity',
        'for': 'FOR Parity',
        'fpr': 'FPR Parity',
        'fnr': 'FNR Parity',
        'tpr': 'TPR Parity',
        'tnr': 'TNR Parity',
        'npv': 'NPV Parity'
    }

    _significance_disparity_mapping = {
        'ppr_disparity': 'ppr_significance',
        'pprev_disparity': 'pprev_significance',
        'precision_disparity': 'precision_significance',
        'fdr_disparity': 'fdr_significance',
        'for_disparity': 'fnr_significance',
        'fpr_disparity': 'fpr_significance',
        'fnr_disparity': 'fnr_significance',
        'tpr_disparity': 'tpr_significance',
        'tnr_disparity': 'tnr_significance',
        'npv_disparity': 'npv_significance'
    }

    def __init__(self, key_metrics=default_absolute_metrics,
                 key_disparities=default_disparities):
        """
        :param key_metrics: Set default absolute group metrics for all subplots
        :param key_disparities: Set default disparity metrics for all subplots
        """
        self.key_metrics = key_metrics
        self.key_disparities = key_disparities

    @staticmethod
    def _nearest_quartile(x):
        '''
        Return nearest quartile for given value x.
        '''
        rounded = round(x * 4) / 4
        if rounded > x:
            return rounded
        else:
            return rounded + 1 / 4

    @staticmethod
    def _check_brightness(rgb_tuple):
        '''
        Determine the brightness of background color in a plot.

        Adapted from https://trendct.org/2016/01/22/how-to-choose-a-label-color-to-contrast-with-background/
        '''
        r, g, b = rgb_tuple
        return (r * 299 + g * 587 + b * 114) / 1000

    @classmethod
    def _brightness_threshold(cls, rgb_tuple, min_brightness, light_color,
                             dark_color='black'):
        '''
        Determine ideal plot label color (light or dark) based on brightness of
        background color based on a given brightness threshold.

        Adapted from https://trendct.org/2016/01/22/how-to-choose-a-label-color-to-contrast-with-background/
        '''
        if cls._check_brightness(rgb_tuple) > min_brightness:
            return dark_color

        return light_color

    @staticmethod
    def _truncate_colormap(orig_cmap, min_value=0.0, max_value=1.0, num_colors=100):
        '''
        Use only part of a colormap (min_value to max_value) across a given number
        of partitions.

        :param orig_cmap: an existing Matplotlib colormap.
        :param min_value: desired minimum value (0.0 to 1.0) for truncated
            colormap. Default is 0.0.
        :param max_value: desired maximum value (0.0 to 1.0) for truncated
            colormap. Default is 1.0.
        :param num_colors: number of colors to spread colormap gradient across
            before truncating. Default is 100.
        :return: Truncated color map

        Attribution: Adapted from: https://stackoverflow.com/questions/
        18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
        '''
        cmap = plt.get_cmap(orig_cmap)
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b: .2f})'.format(n=cmap.name, a=min_value, b=max_value),
            cmap(np.linspace(min_value, max_value, num_colors)))
        return new_cmap


    @classmethod
    def _locate_ref_group_indices(cls, disparities_table, attribute_name, group_metric,
                                 ref_group_flag='_ref_group_value'):
        """
        Finds relative index (row) of reference group value for a given metric.

        :param disparities_table: a disparity table. Output of bias.get_disparity or
            fairness.get_fairness functions.
        :param attribute_name: the attribute to plot metric against. Must be a column
            in the disparities_table.
        :param group_metric: the metric to plot. Must be a column in the
            disparities_table.
        :param ref_group_flag: string indicating column indicates reference group
            flag value. Default is '_ref_group_value'.

        :return: Integer indicating relative index of reference group value row.
        """
        df_models = disparities_table.model_id.unique()
        if len(df_models) == 1:
            model_id = df_models[0]
        else:
            raise ValueError('This method requires one and only one model_id in the disparities table. '
                             'Tip: check disparities_table.model_id.unique() should be just one element list.')
        # get absolute metric name from passed group metric (vs. a disparity name)
        abs_metric = group_metric.replace('_disparity', '')

        all_ref_groups = assemble_ref_groups(disparities_table, ref_group_flag)
        ref_group_name = all_ref_groups[attribute_name][abs_metric]

        # get index for row associated with reference group for that model
        ind = list(disparities_table.loc[(disparities_table['attribute_name'] == attribute_name) &
                                         (disparities_table['attribute_value'] == ref_group_name) &
                                         (disparities_table['model_id'] == model_id)].index)

        # there should only ever be one item in list, but JIC, select first
        if len(ind) == 1:
            idx = ind[0]
        else:
            raise ValueError(f"failed to find only one index for the reference "
                             f"group for attribute_name = {attribute_name} and "
                             f"attribute_value of reference = {ref_group_name} "
                             f"and model_id={model_id}")

        relative_ind = disparities_table.index.get_loc(idx)
        return relative_ind, ref_group_name



    @staticmethod
    def iterate_subplots(axs, ncols, rows, ax_col, ax_row):
        ax_col += 1

        if (ax_col >= ncols) and ((ax_col + 1) % ncols) == 1:
            ax_row += 1
            ax_col = 0

        if rows == 1:
            current_subplot = axs[ax_col]

        elif ncols == 1:
            current_subplot = axs[ax_row]
            ax_row += 1
        else:
            current_subplot = axs[ax_row, ax_col]

        return current_subplot, ax_row, ax_col



    @staticmethod
    def generate_axes(ncols, num_metrics, total_plot_width, sharey, hspace=0.25, indiv_height=6):
        rows = math.ceil(num_metrics / ncols)

        if ncols == 1 or (num_metrics % ncols == 0):
            axes_to_remove = 0
        else:
            axes_to_remove = ncols - (num_metrics % ncols)

        if not (0 < rows <= num_metrics):
            raise ValueError(
                "Plot must have at least one row. Please update number of columns"
                " ('ncols') or check that at least one metric is specified in "
                "'metrics'.")
        if not (0 < ncols <= num_metrics):
            raise ValueError(
                "Plot must have at least one column, and no more columns than "
                "subplots. Please update number of columns ('ncols') or check "
                "that at least one metric is specified in 'metrics'.")

        total_plot_width = total_plot_width

        fig, axs = plt.subplots(nrows=rows, ncols=ncols,
                                figsize=(total_plot_width, indiv_height * rows),
                                sharey=sharey,
                                gridspec_kw={'wspace': 0.075, 'hspace': hspace})

        return fig, axs, rows, axes_to_remove



    def multimodel_plot_group_metric(self, group_table, group_metric,
                                     ncols=3, title=True, label_dict=None,
                                     show_figure=True, selected_models=None,
                                     min_group_size = None):
        """
        Plot a single group metric across all attribute groups for multiple models.

        :param group_table: group table. Output of of group.get_crosstabs() or
            bias.get_disparity functions.
        :param group_metric: the metric to plot. Must be a column in the group_table.
        :param ncols: The number of subplots to plot per row visualization
            figure.
            Default is 3.
        :param title: whether to include a title in visualizations. Default is True.
        :param label_dict: optional, dictionary of replacement labels for data.
            Default is None.
        :param show_figure: Whether to show figure (plt.show()). Default is
            True.
        :param selected_models: which models to visualize. Default is all models in group_table.
        :param min_group_size: minimum size for groups to include in visualization
            (as a proportion of total sample).
        :return: A Matplotlib axis
        """
        # requirement: at least two model_id values
        df_models = self._check_multiple_models(group_table, method_table_name='group_table')

        if not selected_models:
            selected_models = df_models

        plot_table = group_table.loc[group_table['model_id'].isin(selected_models)]


        num_metrics = len(df_models)

        total_plot_width = 25

        fig, axs, rows, axes_to_remove = self.generate_axes(ncols=ncols, num_metrics=num_metrics,
                                                       total_plot_width=total_plot_width, sharey=True)

        # set a different distribution to be plotted in each subplot
        ax_col = -1
        ax_row = 0
        col_num = 0

        for model in df_models:

            if plot_table.loc[plot_table['model_id'] == model, group_metric].isnull().all():
                logging.warning(f"Cannot plot metric '{group_metric}', only NaN values.")
                axes_to_remove += 1
                continue

            elif plot_table.loc[plot_table['model_id'] == model, group_metric].isnull().any():
                # determine which group(s) have missing values
                missing = ", ".join(plot_table.loc[(plot_table['model_id'] == model) &
                                                    (plot_table[
                                                         group_metric].isnull()), 'attribute_value'].values.tolist())

                attr = ", ".join(plot_table.loc[(plot_table['model_id'] == model) &
                                                 (plot_table[
                                                      group_metric].isnull()), 'attribute_name'].values.tolist())

                logging.warning(f"Model {model} '{attr}' group '{missing}' value for metric "
                                f"'{group_metric}' is NA, group not included in visualization.")
                plot_table = plot_table.dropna(axis=0, subset=[group_metric])

                model_table = plot_table.loc[plot_table['model_id'] == model]

            else:
                model_table = plot_table.loc[plot_table['model_id'] == model]

            current_subplot, ax_row, ax_col = self.iterate_subplots(axs, ncols, rows, ax_col, ax_row)

            self.plot_group_metric(group_table=model_table, group_metric=group_metric,
                                   ax=current_subplot, ax_lim=None, title=title, label_dict=label_dict,
                                   min_group_size = min_group_size)

            if title:
                current_subplot.set_title(f"{group_metric.upper()} (Model {model})", fontsize=20)

            col_num += 1

        # disable axes not being used
        if axes_to_remove > 0:
            for i in np.arange(axes_to_remove):
                axs[-1, -(i + 1)].axis('off')

        if show_figure:
            plt.show()

        return fig


    def multimodel_plot_fairness_group(self, fairness_table, group_metric,
                               ncols=3, title=True, label_dict=None, show_figure=True,
                               selected_models=None, min_group_size = None):
        """
        Plot a single group metric colored by parity determination across all
        attribute groups for multiple models.

        :param fairness_table: a fairness table. Output of a Fairness.get_fairness
            function.
        :param group_metric: the metric to plot. Must be a column in the group_table.
        :param ncols: The number of subplots to plot per row visualization
            figure.
            Default is 3.
        :param title: whether to include a title in visualizations. Default is True.
        :param label_dict: optional, dictionary of replacement labels for data.
            Default is None.
        :param show_figure: Whether to show figure (plt.show()). Default is
            True.
        :param selected_models: which models to visualize. Default is all models in fairness_table.
        :param min_group_size: minimum size for groups to include in visualization
            (as a proportion of total sample)

        :return: A Matplotlib axis

        """
        parity_list = list(fairness_table.columns[fairness_table.columns.str.contains(' Parity')])
        if len(parity_list) < 1:
            raise ValueError("multimodel_plot_fairness_disparity: No parity determinations found in fairness_table.")

        # requires at least 2 models
        df_models = self._check_multiple_models(fairness_table, method_table_name='fairness_table')

        if not selected_models:
            selected_models = df_models

        plot_table = fairness_table.loc[fairness_table['model_id'].isin(selected_models)]

        num_metrics = len(df_models)

        total_plot_width = 25

        fig, axs, rows, axes_to_remove = self.generate_axes(ncols=ncols, num_metrics=num_metrics,
                                                       total_plot_width=total_plot_width, sharey=True)

        # set a different distribution to be plotted in each subplot
        ax_col = -1
        ax_row = 0
        col_num = 0

        viz_title = \
            f"MODEL COMPARISON: {group_metric.replace('_', ' ').upper()} PARITY"

        for model in df_models:

            if plot_table.loc[plot_table['model_id'] == model, group_metric].isnull().all():
                logging.warning(f"Cannot plot metric '{group_metric}', only NaN values.")
                axes_to_remove += 1
                continue

            elif plot_table.loc[plot_table['model_id'] == model, group_metric].isnull().any():
                # determine which group(s) have missing values
                missing = ", ".join(plot_table.loc[(plot_table['model_id'] == model) &
                                                       (plot_table[
                                                            group_metric].isnull()), 'attribute_value'].values.tolist())

                attr = ", ".join(plot_table.loc[(plot_table['model_id'] == model) &
                                                    (plot_table[
                                                         group_metric].isnull()), 'attribute_name'].values.tolist())

                logging.warning(f"Model {model} '{attr}' group '{missing}' value for metric "
                                f"'{group_metric}' is NA, group not included in visualization.")

                plot_table = plot_table.dropna(axis=0, subset=[group_metric])

                model_table = plot_table.loc[plot_table['model_id'] == model]

            else:
                model_table = plot_table.loc[plot_table['model_id'] == model]

            current_subplot, ax_row, ax_col = self.iterate_subplots(axs, ncols, rows, ax_col, ax_row)

            self.plot_fairness_group(fairness_table=model_table, group_metric=group_metric,
                                   ax=current_subplot, ax_lim=None, title=title, label_dict=label_dict,
                                   min_group_size = min_group_size)

            if title:
                current_subplot.set_title(f"{group_metric.upper()} (Model {model})", fontsize=20)


            col_num += 1

        # disable axes not being used
        if axes_to_remove > 0:
            for i in np.arange(axes_to_remove):
                axs[-1, -(i + 1)].axis('off')

        if title:
            plt.suptitle(f"{viz_title}", fontsize=25, fontweight="bold")

        if show_figure:
            plt.show()

        return fig


    def multimodel_plot_disparity(self, disparity_table, group_metric,
                                  attribute_name, color_mapping=None,
                                  label_dict=None, title=True, show_figure=True,
                                  highlight_fairness=True, selected_models=None,
                                  min_group_size=None, significance_alpha=0.05):
        """
        Create treemaps to compare multiple model values for a single bias
        disparity metric across attribute groups.

        Adapted from https://plot.ly/python/treemaps/,
        https://gist.github.com/gVallverdu/0b446d0061a785c808dbe79262a37eea,
        and https://fcpython.com/visualisation/python-treemaps-squarify-matplotlib

        :param disparity_table: a disparity table. Output of bias.get_disparity or
            fairness.get_fairness function.
        :param group_metric: the metric to plot. Must be a column in the
            disparity_table.
        :param attribute_name: which attribute to plot group_metric across.
        :param color_mapping: matplotlib colormapping for treemap value boxes.
        :param label_dict: optional, dictionary of replacement labels for data.
            Default is None.
        :param title: whether to include a title in visualizations. Default is True.
        :param highlight_fairness: whether to highlight treemaps by disparity
            magnitude, or by related fairness determination.
        :param show_figure: Whether to show figure (plt.show()). Default is
            True.
        :param selected_models: which models to visualize. Default is all models in disparity_table.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in bias metric
            visualization.
        :param significance_alpha: statistical significance level. Used to
            determine visual representation of significance (number of
            asterisks on treemap).

        :return: A Matplotlib figure
        """
        # requires at least 2 models
        df_models = self._check_multiple_models(disparity_table, method_table_name='disparity_table')

        if not selected_models:
            selected_models = df_models

        plot_table = disparity_table.loc[disparity_table['model_id'].isin(selected_models)]

        if group_metric + '_disparity' not in plot_table.columns:
            related_disparity = group_metric

        else:
            related_disparity = group_metric + '_disparity'

        viz_title = \
            f"MODEL COMPARISON: {related_disparity.replace('_', ' ').upper()}"

        num_metrics = len(df_models)
        ncols=3
        total_plot_width = 25

        fig, axs, rows, axes_to_remove = self.generate_axes(
            ncols=ncols, num_metrics=num_metrics, total_plot_width=total_plot_width,
            sharey=True, hspace=0.5, indiv_height=8)

        # set a different distribution to be plotted in each subplot
        ax_col = -1
        ax_row = 0
        col_num = 0

        for model in df_models:
            model_table = plot_table.loc[plot_table['model_id'] == model]

            current_subplot, ax_row, ax_col = self.iterate_subplots(axs, ncols, rows, ax_col, ax_row)

            self.plot_disparity(model_table, group_metric=group_metric,
                                attribute_name=attribute_name, color_mapping=color_mapping,
                                ax=current_subplot, fig=fig, label_dict=label_dict,
                                title=title, highlight_fairness=highlight_fairness,
                                min_group_size=min_group_size,
                                significance_alpha=significance_alpha)
            if title:
                current_subplot.set_title(f"{related_disparity.replace('_', ' ').upper()}: {attribute_name.upper()} (Model {model})",
                                 fontsize=23)


            col_num += 1

        # disable axes not being used
        if axes_to_remove > 0:
            for i in np.arange(axes_to_remove):
                axs[-1, -(i + 1)].axis('off')

        if title:
            plt.suptitle(f"{viz_title}", fontsize=25, fontweight="bold")

        if show_figure:
            plt.show()

        return fig




    def multimodel_plot_fairness_disparity(self, fairness_table, group_metric,
                                           attribute_name, label_dict=None,
                                           title=True, show_figure=True, selected_models=None,
                                           min_group_size=None, significance_alpha=0.05):
        """
        Create treemaps to compare multiple model fairness determinations for a
        single bias disparity metric across attribute groups.

        :param fairness_table: a fairness table. Output of a Fairness.get_fairness
            function.
        :param group_metric: the metric to plot. Must be a column in the
            disparity_table.
        :param attribute_name: which attribute to plot group_metric across.
        :param color_mapping: matplotlib colormapping for treemap value boxes.
        :param label_dict: optional, dictionary of replacement labels for data.
            Default is None.
        :param title: whether to include a title in visualizations. Default is True.
        :param show_figure: Whether to show figure (plt.show()). Default is
            True.
        :param selected_models: which models to visualize. Default is all models in fairness_table.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in bias metric
            visualization.
        :param significance_alpha: statistical significance level. Used to
            determine visual representation of significance (number of
            asterisks on treemap).

        :return: A Matplotlib figure
        """
        return self.multimodel_plot_disparity(
            disparity_table=fairness_table, group_metric=group_metric,
            attribute_name=attribute_name, label_dict=label_dict, title=title,
            show_figure=show_figure, selected_models=selected_models,
            highlight_fairness=True, min_group_size=min_group_size,
            significance_alpha=significance_alpha)


    @classmethod
    def _check_model_id(cls, df, method_table_name):
        """
        Ensure single model in df, return model_id if so
        """
        if 'model_id' in df.columns:
            df_models = df.model_id.unique()
            if len(df_models) != 1:
                raise ValueError('This method requires one and only one model_id in the dataframe. '
                                 f'Tip: Check that {method_table_name}.model_id.unique() returns a one-element array. ')
            else:
                return df_models[0]
        else:
            return 0

    @classmethod
    def _check_multiple_models(cls, df, method_table_name):
        """
        Ensure multiple models in df, return model_ids if so
        """
        if 'model_id' in df.columns:
            df_models = df.model_id.unique()
            if len(df_models) < 2:
                raise ValueError("This method requires at least two distinct 'model_id' values "
                                 f"in the dataframe. Tip: Check that "
                                 f"{method_table_name}.model_id.unique() returns more than one element.")
            else:
                return df_models
        else:
            raise ValueError("This method requires at least two distinct 'model_id' values "
                             f"in the dataframe. Tip: Check that a 'model_id column exists in "
                             f"'{method_table_name}'.")



    def plot_group_metric(self, group_table, group_metric, ax=None, ax_lim=None,
                          title=True, label_dict=None, min_group_size = None):
        """
        Plot a single group metric across all attribute groups.

        :param group_table: group table. Output of of group.get_crosstabs() or
            bias.get_disparity functions.
        :param group_metric: the metric to plot. Must be a column in the group_table.
        :param ax: a matplotlib Axis. If not passed, a new figure will be created.
        :param title: whether to include a title in visualizations. Default is True.
        :param label_dict: optional, dictionary of replacement labels for data.
            Default is None.
        :param min_group_size: minimum size for groups to include in visualization
            (as a proportion of total sample)

        :return: A Matplotlib axis
        """
        model_id = self._check_model_id(df=group_table, method_table_name='group_table')

        if group_metric not in group_table.columns:
            raise ValueError(f"Specified disparity metric '{group_metric}' not "
                             f"in 'group_table'.")

        if group_table[group_metric].isnull().all():
            raise ValueError(f"Cannot plot {group_metric}, has NaN values.")

        if ax is None:
            (_fig, ax) = plt.subplots(figsize=(10, 5))

        height_of_bar = 1
        attribute_names = group_table.attribute_name.unique()
        tick_indices = []
        next_bar_height = 0

        if min_group_size:
            if min_group_size > (group_table.group_size.max() / group_table.group_size.sum()):
                raise ValueError(f"'min_group_size' proportion specified: '{min_group_size}' "
                                f"is larger than all groups in sample.")

            min_size = min_group_size * group_table.group_size.sum()
            group_table = group_table.loc[group_table['group_size'] >= min_size]

        label_position_values = collections.deque(group_table[group_metric].values)

        lighter_coppers = self._truncate_colormap('copper_r', min_value=0,
                                                   max_value=0.65)

        norm = matplotlib.colors.Normalize(vmin=group_table['group_size'].min(),
                                vmax=group_table['group_size'].max())
        mapping = matplotlib.cm.ScalarMappable(norm=norm, cmap=lighter_coppers)

        # Lock absolute value metric plot x-axis to (0, 1)
        if not ax_lim:
            ax_lim = 1
        ax.set_xlim(0, ax_lim)

        for attribute_name in attribute_names:

            attribute_data = group_table.loc[
                (group_table['attribute_name'] == attribute_name)]

            values = attribute_data[group_metric].values
            grp_sizes = attribute_data['group_size'].values

            attribute_indices = np.arange(next_bar_height,
                                          next_bar_height + attribute_data.shape[0],
                                          step=height_of_bar)
            attribute_tick_location = float((min(attribute_indices) + max(attribute_indices) + height_of_bar)) / 2

            h_attribute = ax.barh(attribute_indices,
                                  width=values,
                                  # label=list(attribute_data['attribute_value'].values),
                                  align='edge', edgecolor='grey')

            label_colors = []
            min_brightness = 0.55

            for bar, g_size in zip(h_attribute, grp_sizes):
                my_col = mapping.to_rgba(g_size)
                bar.set_color(my_col)
                label_colors.append(self._brightness_threshold(
                    rgb_tuple=my_col[:3], min_brightness=min_brightness,
                    light_color=(1, 1, 1, 1)))

            if label_dict:
                labels = [label_dict.get(label, label) for label in
                          attribute_data['attribute_value'].values]
            else:
                labels = attribute_data['attribute_value'].values

            for y, label, value, text_color, g_size in zip(attribute_indices, labels,
                                                   values, label_colors,
                                                   grp_sizes):
                next_position = label_position_values.popleft()
                group_label = f"{label} (Num: {g_size:,})"

                if ax_lim < 3:
                    CHAR_PLACEHOLDER = 0.03
                else:
                    CHAR_PLACEHOLDER = 0.05

                label_length = len(group_label) * CHAR_PLACEHOLDER
                max_val_length = 7 * CHAR_PLACEHOLDER
                indent_length = ax_lim * 0.025

                # bar long enough for label, enough space after bar for value
                if ((indent_length + label_length) < (next_position - indent_length)) and (
                        (next_position + indent_length + max_val_length) < (
                        ax_lim - indent_length)):

                    ax.text(next_position + indent_length, y + float(height_of_bar) / 2,
                            f"{value:.2f}", fontsize=12, verticalalignment='top')
                    ax.text(indent_length, y + float(height_of_bar) / 2,
                            group_label, fontsize=11, verticalalignment='top',
                            color=text_color)

                # case when bar too long for labels after bar, print all text in bar
                elif (next_position + indent_length + max_val_length) > (
                        ax_lim - indent_length):

                    ax.text(indent_length, y + float(height_of_bar) / 2,
                            f"{group_label}, {value:.2f}", fontsize=11,
                            verticalalignment='top', color=text_color)

                # case when bar too small for labels inside bar, print after bar
                else:
                    ax.text(next_position + indent_length, y + float(
                        height_of_bar) / 2,
                            f"{group_label}, {value:.2f}", fontsize=12,
                            verticalalignment='top')

            tick_indices.append((attribute_name, attribute_tick_location))
            next_bar_height = max(attribute_indices) + 2 * height_of_bar

        ax.yaxis.set_ticks(list(map(lambda x: x[1], tick_indices)))
        ax.yaxis.set_ticklabels(list(map(lambda x: x[0], tick_indices)), fontsize=14)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='lightgray', which='major',linestyle='dashed')
        ax.set_xlabel("Absolute Metric Magnitude")

        if title:
            ax.set_title(f"{group_metric.upper()} (Model {model_id})", fontsize=20)

        return ax


    def plot_disparity(self, disparity_table, group_metric, attribute_name,
                       color_mapping=None, ax=None, fig=None,
                       label_dict=None, title=True,
                       highlight_fairness=False, min_group_size=None,
                       significance_alpha=0.05):
        """
        Create treemap based on a single bias disparity metric across attribute
        groups.

        Adapted from https://plot.ly/python/treemaps/,
        https://gist.github.com/gVallverdu/0b446d0061a785c808dbe79262a37eea,
        and https://fcpython.com/visualisation/python-treemaps-squarify-matplotlib

        :param disparity_table: a disparity table. Output of bias.get_disparity or
            fairness.get_fairness function.
        :param group_metric: the metric to plot. Must be a column in the
            disparity_table.
        :param attribute_name: which attribute to plot group_metric across.
        :param color_mapping: matplotlib colormapping for treemap value boxes.
        :param ax: a matplotlib Axis. If not passed, a new figure will be created.
        :param fig: a matplotlib Figure. If not passed, a new figure will be created.
        :param label_dict: optional, dictionary of replacement labels for data.
            Default is None.
        :param title: whether to include a title in visualizations. Default is True.
        :param highlight_fairness: whether to highlight treemaps by disparity
            magnitude, or by related fairness determination.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in bias metric
            visualization
        :param significance_alpha: statistical significance level. Used to
            determine visual representation of significance (number of
            asterisks on treemap).

        :return: A Matplotlib axis
        """
        # Use matplotlib to truncate colormap, scale metric values
        # between the min and max, then assign colors to individual values

        model_id = self._check_model_id(df=disparity_table, method_table_name='disparities_table')

        table_columns = set(disparity_table.columns)
        if group_metric not in table_columns:
           raise ValueError(f"Specified disparity metric {group_metric} not in 'disparity_table'.")


        attribute_table = \
            disparity_table.loc[disparity_table['attribute_name'] == attribute_name]

        # sort by group size, as box size is indicative of group size
        sorted_df = attribute_table.sort_values('group_size', ascending=False)

        x = 0.
        y = 0.
        width = 100.
        height = 100.

        ref_group_rel_idx, ref_group_name = \
            self._locate_ref_group_indices(disparities_table=sorted_df,
                                            attribute_name=attribute_name,
                                            group_metric=group_metric)

        if min_group_size:
            if min_group_size > (disparity_table.group_size.max() /
                            disparity_table.group_size.sum()):
                raise ValueError(f"'min_group_size' proportion specified: '{min_group_size}' "
                                f"is larger than all groups in sample.")

            min_size = min_group_size * disparity_table.group_size.sum()

            # raise warning if minimum group size specified would exclude
            # reference group
            if any(sorted_df.loc[(sorted_df['attribute_value']==ref_group_name),
                                 ['group_size']].values < min_size):
                logging.warning(
                    f"Reference group size is smaller than 'min_group_size' proportion "
                    f"specified: '{min_group_size}'. Reference group '{ref_group_name}' "
                    f"was not excluded.")

            sorted_df = \
                    sorted_df.loc[(sorted_df['group_size'] >= min_size) |
                                  (sorted_df['attribute_value'] == ref_group_name)]

        # select group size as values for size of boxes
        values = sorted_df.loc[:, 'group_size']

        # get new index for ref group
        ref_group_rel_idx, _ = \
            self._locate_ref_group_indices(disparities_table=sorted_df,
                                            attribute_name=attribute_name,
                                            group_metric=group_metric)

        # labels for squares in tree map:
        # label should always be disparity value (but boxes visualized should be
        # always be the metric absolute value capped between 0.1x ref group and
        # 10x ref group)
        if group_metric + '_disparity' not in attribute_table.columns:
            related_disparity = group_metric

        else:
            related_disparity = group_metric + '_disparity'


        if highlight_fairness:
            if not len(table_columns.intersection(self._metric_parity_mapping.values())) > 1:
                raise ValueError("Data table must include at least one fairness "
                                 "determination to visualize metric parity.")

            # apply red for "False" fairness determinations and green for "True"
            # determinations
            cb_green = '#1b7837'
            cb_red = '#a50026'

            parity = self._metric_parity_mapping[group_metric]
            if (parity not in table_columns):
                raise ValueError(
                    f"Related fairness determination for {group_metric} must be "
                    f"included in data table to color visualization based on "
                    f"metric fairness.")
            clrs = [cb_green if val else cb_red for val in sorted_df[parity]]

        else:
            aq_palette = sns.diverging_palette(225, 35, sep=10, as_cmap=True)

            if not color_mapping:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
                color_mapping = matplotlib.cm.ScalarMappable(norm=norm, cmap=aq_palette)

            clrs = \
                [color_mapping.to_rgba(val) for val in sorted_df[related_disparity]]

        # color reference group grey
        clrs[ref_group_rel_idx] = '#D3D3D3'

        compare_value = values.iloc[ref_group_rel_idx]

        scaled_values = [(0.1 * compare_value) if val < (0.1 * compare_value) else
                         (10 * compare_value) if val >= (10 * compare_value) else
                         val for val in values]

        label_values = \
            ["(Ref)" if attr_val == ref_group_name else
             f"{disp:.2f}" for attr_val, disp in
             zip(sorted_df['attribute_value'], sorted_df[related_disparity]) ]

        if label_dict:
            labels = \
                [label_dict.get(label, label) for label in sorted_df['attribute_value']]
        else:
            labels = sorted_df['attribute_value'].values

        # if df includes significance columns, add stars to indicate significance
        if self._significance_disparity_mapping[related_disparity] in sorted_df.columns:

            # truncated_signif_mapping = {k: v for k,v in self._significance_disparity_mapping.items() if v in sorted_df.columns}

            if sorted_df.columns[
                sorted_df.columns.str.contains('_significance')].value_counts().sum() > 0:

            # unmasked significance
            # find indices where related significance have smaller value than significance_alpha
                if np.issubdtype(
                        sorted_df[
                            self._significance_disparity_mapping[related_disparity]].dtype,
                            # truncated_signif_mapping[related_disparity]].dtype,
                        np.number):
                    to_star = sorted_df.loc[
                        sorted_df[
                            self._significance_disparity_mapping[related_disparity]] < significance_alpha].index.tolist()
                            # truncated_signif_mapping[related_disparity]] < significance_alpha].index.tolist()


                # masked significance
                # find indices where attr values have True value for each of those two columns,
                else:
                    to_star = sorted_df.loc[
                        sorted_df[
                            self._significance_disparity_mapping[related_disparity]] > 0].index.tolist()
                            # truncated_signif_mapping[related_disparity]] > 0].index.tolist()


                # add stars to label value where significant
                for idx in to_star:
                    # convert idx location to relative index in sorted df and label_values list
                    idx_adj = sorted_df.index.get_loc(idx)

                    # star significant disparities in visualizations based on significance level
                    if 0.10 >= significance_alpha > 0.05:
                        significance_stars = '*'
                    elif 0.05 >= significance_alpha > 0.01:
                        significance_stars = '**'
                    elif significance_alpha <= 0.01:
                        significance_stars = '***'
                    else:
                        significance_stars = ''
                    label_values[idx_adj] = label_values[idx_adj] + significance_stars


        normed = sf.normalize_sizes(scaled_values, width, height)

        padded_rects = sf.padded_squarify(normed, x, y, width, height)

        # make plot
        if not (ax and fig):
            fig, ax = plt.subplots(figsize=(5, 4))

        ax = sf.squarify_plot_rects(padded_rects, color=clrs, labels=labels,
                                    values=label_values, ax=ax, alpha=0.8,
                                    acronyms=False)

        if title:
            ax.set_title(f"{related_disparity.replace('_', ' ').upper()}: {attribute_name.upper()}",
                     fontsize=23)

        if not highlight_fairness:
            # create dummy invisible image with a color map to leverage for color bar
            img = plt.imshow([[0, 2]], cmap=aq_palette, alpha=0.8)
            img.set_visible(False)
            fig.colorbar(img, orientation="vertical", shrink=.96, ax=ax)

        #     Remove axes and display the plot
        ax.axis('off')


    def plot_fairness_group(self, fairness_table, group_metric, ax=None,
                            ax_lim=None, title=False, label_dict=None,
                            min_group_size=None):
        '''
        This function plots absolute group metrics as indicated by the config file,
        colored based on calculated parity.

        :param fairness_table: fairness table. Output of a Fairness.get_fairness
            function.
        :param group_metric: the fairness metric to plot. Must be a column in the fairness_table.
        :param ax: a matplotlib Axis. If not passed a new figure will be created.
        :param ax_lim: maximum value on x-axis, used to match axes across subplots
            when plotting multiple metrics. Default is None.
        :param title: whether to include a title in visualizations. Default is True.
        :param label_dict: optional dictionary of replacement values for data.
            Default is None.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in fairness
            visualization

        :return: A Matplotlib axis
        '''
        model_id = self._check_model_id(df=fairness_table, method_table_name='fairness_table')

        if group_metric not in fairness_table.columns:
            raise ValueError(f"Specified disparity metric {group_metric} not "
                             f"in 'fairness_table'.")

        if fairness_table[group_metric].isnull().any():
            raise ValueError(f"Cannot plot {group_metric}, has NaN values.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        height_of_bar = 1
        attributes = fairness_table.attribute_name.unique()
        tick_indices = []
        next_bar_height = 0

        if min_group_size:
            if min_group_size > (fairness_table.group_size.max() / fairness_table.group_size.sum()):
                raise ValueError(f"'min_group_size' proportion specified: '{min_group_size}' "
                                f"is larger than all groups in sample.")

            min_size = min_group_size * fairness_table.group_size.sum()
            fairness_table = fairness_table.loc[fairness_table['group_size'] >= min_size]

        label_position_values = collections.deque(fairness_table[group_metric].values)


        # Lock absolute value metric plot x-axis to (0, 1)
        if not ax_lim:
            ax_lim = 1
        ax.set_xlim(0, ax_lim)

        for attribute in attributes:
            attribute_data = fairness_table.loc[
                fairness_table['attribute_name'] == attribute]
            values = attribute_data[group_metric].values
            grp_sizes = attribute_data['group_size'].values

            # apply red for "False" fairness determinations and green for "True"
            # determinations
            cb_green = '#1b7837'
            cb_red = '#a50026'
            parity = self._metric_parity_mapping[group_metric]
            parity_colors = [cb_green if val else
                              cb_red for val in attribute_data[parity]]

            # Set white text for red bars and black text for green bars
            label_colors = [(0, 0, 0, 1) if val == True else
                            (1, 1, 1, 1) for val in attribute_data[parity]]

            attribute_indices = \
                np.arange(next_bar_height, next_bar_height + attribute_data.shape[0],
                          step=height_of_bar)

            attribute_tick_location = \
                float((min(attribute_indices) + max(attribute_indices) +
                       height_of_bar)) / 2

            h_attribute = ax.barh(attribute_indices,
                                  width=values,
                                  color=parity_colors,
                                  align='edge', edgecolor='grey', alpha=0.8)

            if label_dict:
                labels = [label_dict.get(label, label) for label in
                          attribute_data['attribute_value'].values]
            else:
                labels = attribute_data['attribute_value'].values

            for y, label, value, text_color, g_size in zip(
                    attribute_indices, labels, values, label_colors,
                    grp_sizes):

                next_position = label_position_values.popleft()
                group_label = f"{label} (Num: {g_size:,})"

                if ax_lim < 3:
                    CHAR_PLACEHOLDER = 0.03
                else:
                    CHAR_PLACEHOLDER = 0.25

                label_length = len(group_label) * CHAR_PLACEHOLDER
                max_val_length = 7 * CHAR_PLACEHOLDER
                indent_length = ax_lim * 0.025

                # bar long enough for label, enough space after bar for value
                if ((indent_length + label_length) < (next_position - indent_length)) and (
                        (next_position + indent_length + max_val_length) < (
                        ax_lim - indent_length)):

                    ax.text(next_position + indent_length, y + float(height_of_bar) / 2,
                            f"{value:.2f}", fontsize=12, verticalalignment='top')
                    ax.text(indent_length, y + float(height_of_bar) / 2,
                            group_label, fontsize=11, verticalalignment='top',
                            color=text_color)

                # case when bar too long for labels after bar, print all text in bar
                elif (next_position + indent_length + max_val_length) > (
                        ax_lim - indent_length):

                    ax.text(indent_length, y + float(height_of_bar) / 2,
                            f"{group_label}, {value:.2f}", fontsize=11,
                            verticalalignment='top', color=text_color)

                # case when bar too small for labels inside bar, print all text
                # after bar
                else:
                    ax.text(next_position + indent_length,
                            y + float(height_of_bar) / 2,
                            f"{group_label}, {value:.2f}", fontsize=12,
                            verticalalignment='top')

            tick_indices.append((attribute, attribute_tick_location))
            next_bar_height = max(attribute_indices) + 2 * height_of_bar

        ax.yaxis.set_ticks(list(map(lambda x: x[1], tick_indices)))
        ax.yaxis.set_ticklabels(list(map(lambda x: x[0], tick_indices)), fontsize=14)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='lightgray', which='major', linestyle='dashed')

        ax.set_xlabel('Absolute Metric Magnitude')

        if title:
            ax.set_title(f"{group_metric.upper()}", fontsize=20)

        return ax



    def plot_fairness_disparity(self, fairness_table, group_metric,
                                attribute_name, ax=None, fig=None,
                                title=True, min_group_size=None,
                                significance_alpha=0.05):
        """
        Plot disparity metrics colored based on calculated disparity.

        :param group_metric: the metric to plot. Must be a column in the disparity_table.
        :param attribute_name: which attribute to plot group_metric across.
        :param ax: a matplotlib Axis. If not passed, a new figure will be created.
        :param fig: a matplotlib Figure. If not passed, a new figure will be created.
        :param title: whether to include a title in visualizations. Default is True.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in bias metric
            visualization
        :param significance_alpha: statistical significance level. Used to
            determine visual representation of significance (number of
            asterisks on treemap).
        :return: A Matplotlib axis
        """
        return self.plot_disparity(disparity_table=fairness_table,
                                   group_metric=group_metric,
                                   attribute_name=attribute_name,
                                   color_mapping=None,
                                   ax=ax, fig=fig, highlight_fairness=True,
                                   min_group_size=min_group_size, title=title,
                                   significance_alpha=significance_alpha)

    def _plot_multiple(self, data_table, plot_fcn, metrics=None, fillzeros=True,
                        title=True, ncols=3, label_dict=None, show_figure=True,
                        min_group_size=None):
        """
        This function plots bar charts of absolute metrics indicated by config
        file

        :param data_table: output of group.get_crosstabs, bias.get_disparity, or
            fairness.get_fairness functions
        :param plot_fcn: the single-metric plotting function to use for subplots
        :param metrics: which metric(s) to plot, or 'all.' If this value is
            null, will plot the following absolute metrics (or related disparity
            measures):
                - Predicted Prevalence (pprev),
                - Predicted Positive Rate (ppr),
                - False Discovery Rate (fdr),
                - False Omission Rate (for),
                - False Positive Rate (fpr),
                - False Negative Rate (fnr)
        :param fillzeros: Should null values be filled with zeros. Default is
            True.
        :param title: Whether to display a title on each plot. Default is True.
        :param ncols: The number of subplots to plot per row visualization
            figure.
            Default is 3.
        :param label_dict: Optional dictionary of label replacements. Default is
            None.
        :param show_figure: Whether to show figure (plt.show()). Default is
            True.
        :param min_group_size: Minimum proportion of total group size (all data)
            a population group must meet in order to be included in visualization

        :return: Returns a figure
        """
        model_id = self._check_model_id(df=data_table, method_table_name='data_table')

        if fillzeros:
            data_table = data_table.fillna(0)

        if plot_fcn in [self.plot_fairness_group, self.plot_group_metric]:
            if not metrics:
                metrics = \
                    [met for met in self.key_metrics if met in data_table.columns]

            elif metrics == 'all':
                all_abs_metrics = ('pprev', 'ppr', 'fdr', 'for', 'fpr', 'fnr',
                               'tpr', 'tnr', 'npv', 'precision')
                metrics = \
                    [met for met in all_abs_metrics if met in data_table.columns]

            ax_lim = 1

        # elif plot_fcn in [self.plot_fairness_disparity, self.plot_disparity]:
        else:
            if not metrics:
                metrics = \
                    [disp for disp in self.key_disparities if disp in data_table.columns]
            elif metrics == 'all':
                metrics = \
                    list(data_table.columns[data_table.columns.str.contains('_disparity')])

            ax_lim = min(10, self._nearest_quartile(max(data_table[metrics].max())) + 0.1)

        num_metrics = len(metrics)
        rows = math.ceil(num_metrics / ncols)
        if ncols == 1 or (num_metrics % ncols == 0):
            axes_to_remove = 0
        else:
            axes_to_remove = ncols - (num_metrics % ncols)

        if not (0 < rows <= num_metrics):
           raise ValueError (
               "Plot must have at least one row. Please update number of columns"
               " ('ncols') or check that at least one metric is specified in "
               "'metrics'.")
        if not (0 < ncols <= num_metrics):
           raise ValueError(
               "Plot must have at least one column, and no more columns than "
               "subplots. Please update number of columns ('ncols') or check "
               "that at least one metric is specified in 'metrics'.")

        total_plot_width = 25

        fig, axs = plt.subplots(nrows=rows, ncols=ncols,
                                figsize=(total_plot_width, 6 * rows),
                                sharey=True,
                                gridspec_kw={'wspace': 0.075, 'hspace': 0.25})

        # set a different metric to be plotted in each subplot
        ax_col = 0
        ax_row = 0

        viz_title = \
            f"{(', ').join(list(map(lambda x: x.upper(), metrics)))}"

        for group_metric in metrics:
            if (ax_col >= ncols) and ((ax_col + 1) % ncols) == 1:
                ax_row += 1
                ax_col = 0

            if rows == 1:
                current_subplot = axs[ax_col]

            elif ncols == 1:
                current_subplot = axs[ax_row]
                ax_row += 1
            else:
                current_subplot = axs[ax_row, ax_col]

            plot_fcn(data_table, group_metric=group_metric, ax=current_subplot,
                     ax_lim=ax_lim, title=title, label_dict=label_dict,
                     min_group_size=min_group_size)
            ax_col += 1

        # disable axes not being used
        if axes_to_remove > 0:
            for i in np.arange(axes_to_remove):
                axs[-1, -(i + 1)].axis('off')

        if show_figure:
            plt.show()

        if title:
            plt.suptitle(f"{viz_title}", fontsize=25, fontweight="bold")

        return fig


    def _plot_multiple_treemaps(self, data_table, plot_fcn, attributes=None,
                                 metrics=None, fillzeros=True, title=True,
                                 label_dict=None, highlight_fairness=False,
                                 show_figure=True, min_group_size=None,
                                 significance_alpha=0.05):
        """
        This function plots treemaps of disparities indicated by config file

        :param data_table: output of bias.get_disparity, or fairness.get_fairness
            functions
        :param plot_fcn: Plotting function to use to plot individual disparity
            or fairness treemaps in grid
        :param attributes: which attributes to plot against. Must be specified
            if no metrics specified.
        :param metrics: which metric(s) to plot, or 'all.' MUST be specified if
            no attributes specified. If this value is null, the following
            absolute metrics/ related disparity measures will be plotted against
            specified attributes:
                - Predicted Prevalence (pprev),
                - Predicted Positive Rate (ppr),
                - False Discovery Rate (fdr),
                - False Omission Rate (for),
                - False Positive Rate (fpr),
                - False Negative Rate (fnr)
        :param fillzeros: Whether null values should be filled with zeros. Default
            is True.
        :param title: Whether to display a title on each plot. Default is True.
        :param label_dict: Optional dictionary of label replacements. Default is
            None.
        :param highlight_fairness: Whether to highlight treemaps by disparity
            magnitude, or by related fairness determination.
        :param show_figure: Whether to show figure (plt.show()). Default is True.
        :param min_group_size: Minimum proportion of total group size (all data)
            a population group must meet in order to be included in visualization
        :param significance_alpha: statistical significance level. Used to
            determine visual representation of significance (number of
            asterisks on treemap).

        :return: Returns a figure
        """
        model_id = self._check_model_id(df=data_table, method_table_name='data_table')

        if fillzeros:
            data_table = data_table.fillna(0)

        if all(v is None for v in [attributes, metrics]):
            raise ValueError("One of the following parameters must be specified: " \
                        "'attribute', 'metrics'.")

        if attributes:
            if not metrics:
                metrics = [abs_m for abs_m in self.key_metrics if
                           abs_m in data_table.columns]

            #         metrics = list(set(self.input_group_metrics) &
            # set(data_table.columns))
            elif metrics == 'all':
                all_abs_metrics = ['tpr_disparity', 'tnr_disparity', 'for_disparity',
                               'fdr_disparity', 'fpr_disparity', 'fnr_disparity',
                               'npv_disparity', 'precision_disparity',
                               'ppr_disparity', 'pprev_disparity']
                metrics = \
                    [abs_m for abs_m in all_abs_metrics if abs_m in data_table.columns]

            viz_title = \
                f"DISPARITY METRICS BY {(', ').join(list(map(lambda x:x.upper(), attributes)))}"

        elif not attributes:
            attributes = list(data_table.attribute_name.unique())
            if metrics == 'all':
                all_disparities = ['tpr_disparity', 'tnr_disparity', 'for_disparity',
                               'fdr_disparity', 'fpr_disparity', 'fnr_disparity',
                               'npv_disparity', 'precision_disparity',
                               'ppr_disparity', 'pprev_disparity']
                metrics = [disparity for disparity in all_disparities if
                           disparity in data_table.columns]
            viz_title = f"{(', ').join(map(lambda x:x.upper(), metrics))} " \
                        f"ACROSS ATTRIBUTES"

        num_metrics = len(attributes) * len(metrics)
        if num_metrics > 2:
            ncols = 3
        else:
            ncols = num_metrics

        rows = math.ceil(num_metrics / ncols)
        if ncols == 1 or (num_metrics % ncols == 0):
            axes_to_remove = 0
        else:
            axes_to_remove = ncols - (num_metrics % ncols)

        if not (0 < rows <= num_metrics):
           raise ValueError (
               "Plot must have at least one row. Please update number of columns"
               " ('ncols'), the list of metrics to be plotted ('metrics'), or "
               "the list of attributes to plot disparity metrics across.")
        if not (0 < ncols <= num_metrics):
           raise ValueError(
               "Plot must have at least one column, and no more columns than "
               "plots. Please update number of columns ('ncols'), the list of "
               "metrics to be plotted ('metrics'), or the list of attributes to "
               "plot disparity metrics across.")

        total_plot_width = 25

        fig, axs = plt.subplots(nrows=rows, ncols=ncols,
                                figsize=(total_plot_width, 8 * rows),
                                gridspec_kw={'wspace': 0.025, 'hspace': 0.5},
                                subplot_kw={'aspect': 'equal'})

        if highlight_fairness:
            mapping = None
        else:
            aq_palette = sns.diverging_palette(225, 35, sep=10, as_cmap=True)

            norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
            mapping = matplotlib.cm.ScalarMappable(norm=norm, cmap=aq_palette)

        # set a different metric to be plotted in each subplot
        ax_col = 0
        ax_row = 0


        for group_metric in metrics:
            for attr in attributes:
                if (ax_col >= ncols) and ((ax_col + 1) % ncols) == 1:
                    ax_row += 1
                    ax_col = 0

                if num_metrics == 1:
                    current_subplot = axs

                elif (num_metrics > 1) and (rows == 1):
                    current_subplot = axs[ax_col]

                elif (num_metrics > 1) and (ncols == 1):
                    current_subplot = axs[ax_row]
                    ax_row += 1
                else:
                    current_subplot = axs[ax_row, ax_col]

                plot_fcn(data_table, group_metric=group_metric,
                         attribute_name=attr, color_mapping=mapping,
                         ax=current_subplot, fig=fig, title=title,
                         label_dict=label_dict,
                         highlight_fairness=highlight_fairness,
                         min_group_size=min_group_size, significance_alpha=significance_alpha)

                ax_col += 1

        # disable axes not being used
        if axes_to_remove > 0:
            for i in np.arange(axes_to_remove):
                axs[-1, -(i + 1)].axis('off')

        plt.suptitle(f"{viz_title}", fontsize=25, fontweight="bold")

        # fig.tight_layout()

        if rows > 2:
            fig.subplots_adjust(top=0.95)
        else:
            fig.subplots_adjust(top=0.90)

        if show_figure:
            plt.show()
        return fig



    def plot_group_metric_all(self, data_table, metrics=None, fillzeros=True,
                              ncols=3, title=True, label_dict=None,
                              show_figure=True, min_group_size=None):
        """
        Plot multiple metrics at once from a fairness object table.

        :param data_table:  output of group.get_crosstabs, bias.get_disparity, or
            fairness.get_fairness functions.
        :param metrics: which metric(s) to plot, or 'all.'
            If this value is null, will plot:
                - Predicted Prevalence (pprev),
                - Predicted Positive Rate (ppr),
                - False Discovery Rate (fdr),
                - False Omission Rate (for),
                - False Positive Rate (fpr),
                - False Negative Rate (fnr)
        :param fillzeros: whether to fill null values with zeros. Default is
            True.
        :param ncols: number of subplots per row in figure. Default is 3.
        :param title: whether to display a title on each plot. Default is True.
        :param label_dict: optional dictionary of label replacements. Default is None.
        :param show_figure: whether to show figure (plt.show()). Default is True.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in group metric
            visualization.

        :return: A Matplotlib figure
        """
        return self._plot_multiple(
            data_table, plot_fcn=self.plot_group_metric, metrics=metrics,
            fillzeros=fillzeros, title=title, ncols=ncols, label_dict=label_dict,
            show_figure=show_figure, min_group_size=min_group_size)


    def plot_disparity_all(self, data_table, attributes=None, metrics=None,
                           fillzeros=True, title=True, label_dict=None,
                           show_figure=True, min_group_size=None,
                           significance_alpha=0.05):
        """
        Plot multiple metrics at once from a fairness object table.

        :param data_table:  output of group.get_crosstabs, bias.get_disparity, or
            fairness.get_fairness functions.
        :param attributes: which attribute(s) to plot metrics for. If this
            value is null, will plot metrics against all attributes.
        :param metrics: which metric(s) to plot, or 'all.'
            If this value is null, will plot:
                - Predicted Prevalence Disparity (pprev_disparity),
                - Predicted Positive Rate Disparity (ppr_disparity),
                - False Discovery Rate Disparity (fdr_disparity),
                - False Omission Rate Disparity (for_disparity),
                - False Positive Rate Disparity (fpr_disparity),
                - False Negative Rate Disparity (fnr_disparity)
        :param fillzeros: whether to fill null values with zeros. Default is True.
        :param title: whether to display a title on each plot. Default is True.
        :param label_dict: optional dictionary of label replacements. Default is
            None.
        :param show_figure: whether to show figure (plt.show()). Default is True.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in metric
            visualization.
        :param significance_alpha: statistical significance level. Used to
            determine visual representation of significance (number of
            asterisks on treemap).

        :return: A Matplotlib figure
        """
        return self._plot_multiple_treemaps(
            data_table, plot_fcn=self.plot_disparity, attributes=attributes,
            metrics=metrics, fillzeros=fillzeros, label_dict=label_dict,
            highlight_fairness=False, show_figure=show_figure, title=title,
            min_group_size=min_group_size, significance_alpha=significance_alpha)


    def plot_fairness_group_all(self, fairness_table, metrics=None, fillzeros=True,
                                ncols=3, title=True, label_dict=None,
                                show_figure=True, min_group_size=None):
        """
        Plot multiple metrics at once from a fairness object table.

        :param fairness_table: fairness table. Output of a Fairness.get_fairness_
            function.
        :param metrics: which metric(s) to plot, or 'all.'
            If this value is null, will plot:
                - Predicted Prevalence (pprev),
                - Predicted Positive Rate (ppr),
                - False Discovery Rate (fdr),
                - False Omission Rate (for),
                - False Positive Rate (fpr),
                - False Negative Rate (fnr)
        :param fillzeros: whether to fill null values with zeros. Default is True.
        :param ncols: number of subplots per row in figure. Default is 3.
        :param title: whether to display a title on each plot. Default is True.
        :param label_dict: optional dictionary of label replacements. Default is
            None.
        :param show_figure: whether to show figure (plt.show()). Default is True.
        :param min_group_size: minimum proportion of total group size (all data).
            a population group must meet in order to be included in fairness
            visualization

        :return: A Matplotlib figure
        """
        return self._plot_multiple(
            fairness_table, plot_fcn=self.plot_fairness_group, metrics=metrics,
            fillzeros=fillzeros, title=title, ncols=ncols, label_dict=label_dict,
            show_figure=show_figure, min_group_size=min_group_size)

    def plot_fairness_disparity_all(self, fairness_table, attributes=None,
                                    metrics=None, fillzeros=True, title=True,
                                    label_dict=None, show_figure=True,
                                    min_group_size=None, significance_alpha=0.05):
        """
        Plot multiple metrics at once from a fairness object table.

        :param fairness_table: a fairness table. Output of a Fairness.get_fairness
            function.
        :param attributes: which attribute(s) to plot metrics for. If this value is null, will plot metrics against all attributes.
        :param metrics: which metric(s) to plot, or 'all.'
            If this value is null, will plot:
                - Predicted Prevalence Disparity (pprev_disparity),
                - Predicted Positive Rate Disparity (ppr_disparity),
                - False Discovery Rate Disparity (fdr_disparity),
                - False Omission Rate Disparity (for_disparity),
                - False Positive Rate Disparity (fpr_disparity),
                - False Negative Rate Disparity (fnr_disparity)
        :param fillzeros: whether to fill null values with zeros. Default is True.
        :param title: whether to display a title on each plot. Default is True.
        :param label_dict: optional dictionary of label replacements. Default is
            None.
        :param show_figure: whether to show figure (plt.show()). Default is True.
        :param min_group_size: minimum proportion of total group size (all data)
            a population group must meet in order to be included in fairness
            visualization
        :param significance_alpha: statistical significance level. Used to
            determine visual representation of significance (number of
            asterisks on treemap)

        :return: A Matplotlib figure
        """
        return self._plot_multiple_treemaps(
            fairness_table, plot_fcn=self.plot_disparity, attributes=attributes,
            metrics=metrics, fillzeros=fillzeros, label_dict=label_dict,
            title=title, highlight_fairness=True, show_figure=show_figure,
            min_group_size=min_group_size, significance_alpha=significance_alpha)


    def multimodel_attribute_comparison(self, disparity_table, attribute, x_metric, y_metric='precision',
                              x_jitter=None, y_jitter=None, selected_models=None, ncols=3,
                              scatter_kws={'legend': 'full'}, title=True, sharey=True,
                              show_figure=True):
        """
        :param disparity_table: disparity table. output of bias.get_disparity, or
            fairness.get_fairness function.
        :param attribute: attributes: which attribute values (sample groups) to plot x and y metrics for.
        :param x_metric: the metric to plot on the X axis. Must be a column in the disparity_table.
        :param y_metric: the metric to plot on the Y axis. Must be a column in the disparity_table.
        :param x_jitter: jitter for x values. Default is None.
        :param y_jitter: jitter for y values. Default is None.
        :param selected_models: which models to visualize. Default is all models in disparity_table.
        :param ncols: The number of subplots to plot per row visualization
            figure.
            Default is 3.
        :param scatter_kws: keyword arguments for scatterplot
        :param title: whether to include a title in visualizations. Default is True.
        :param sharey: whether comparison subplots should share Y axis. Default is True
        :param show_figure: whether to show figure (plt.show()). Default is True.

        :return: A Matplotlib figure
        """

        df_models = self._check_multiple_models(disparity_table, method_table_name='disparity_table')

        if not selected_models:
            selected_models = df_models

        attribute_table = disparity_table.loc[(disparity_table['attribute_name']==attribute) & (disparity_table['model_id'].isin(selected_models))]

        groups = attribute_table.attribute_value.unique()

        num_metrics = len(groups)

        total_plot_width = 25

        fig, axs, rows, axes_to_remove = self.generate_axes(ncols=ncols, num_metrics=num_metrics,
                                                            total_plot_width=total_plot_width,
                                                            sharey=sharey, hspace=0.5,
                                                            indiv_height=6)

        # set a different distribution to be plotted in each subplot
        ax_col = -1
        ax_row = 0
        col_num = 0

        viz_title = \
            f"MODEL COMPARISON: {x_metric.replace('_', ' ').upper()} BY {y_metric.replace('_', ' ').upper()} " \
                f"ACROSS {attribute.replace('_', ' ').upper()}"

        aq_palette = sns.diverging_palette(225, 35, sep=10, n=20, as_cmap=True, center="dark")

        for group in groups:
            # subset df to get only that attribute, no need to aggregate
            group_table = attribute_table.loc[attribute_table['attribute_value'] == group]

            if group_table.loc[group_table['attribute_value'] == group, x_metric].isnull().all():
                logging.warning(f"Cannot plot metric '{x_metric}' for group '{group}', only NaN values."
                                f" Continuing with remaining groups.")
                axes_to_remove += 1
                continue

            elif group_table.loc[group_table['attribute_value'] == group, y_metric].isnull().all():
                logging.warning(f"Cannot plot metric '{y_metric}' for group '{group}', only NaN values. "
                                f"Continuing with remaining groups.")
                axes_to_remove += 1
                continue

            current_subplot, ax_row, ax_col = self.iterate_subplots(axs, ncols, rows, ax_col, ax_row)

            with sns.axes_style("whitegrid"):
                # scatterplot of each model for that atttibute value group
                 sns.scatterplot(x=x_metric, y=y_metric, data=group_table, hue='model_id', palette=aq_palette,
                                 x_jitter=x_jitter, y_jitter=y_jitter, ax=current_subplot, **scatter_kws)

            current_subplot.xaxis.grid(color='lightgray', which='major')
            current_subplot.yaxis.grid(color='lightgray', which='major')
            labels = [item.get_text().replace('_', ' ').upper() for item in current_subplot.get_xticklabels()]
            if '' not in labels:
                current_subplot.set_xticklabels(labels, rotation=30, ha='center')
            else:
                plt.xticks(rotation=30, horizontalalignment='center')

            x_clean = x_metric.replace('_', ' ').upper()
            y_clean = y_metric.replace('_', ' ').upper()
            current_subplot.set_xlabel(x_clean, fontsize=12)
            current_subplot.set_ylabel(y_clean, fontsize=12)

            handles, labels = current_subplot.get_legend_handles_labels()
            current_subplot.legend(handles=handles[1:], labels=[f"Model {model}" for model in labels[1:]], title="Model ID")
            plot_title = f"MODEL COMPARISON:\n{y_clean} BY {x_clean} ({attribute.replace('_',' ').upper()}: {group.replace('_',' ').upper()})"
            current_subplot.set_title(plot_title, fontsize=20)

            # current_subplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            col_num += 1

        # disable axes not being used
        if axes_to_remove > 0:
            for i in np.arange(axes_to_remove):
                axs[-1, -(i + 1)].axis('off')

        if title:
            plt.suptitle(f"{viz_title}", fontsize=25, fontweight="bold")

        if show_figure:
            plt.show()

        return fig



    def multimodel_comparison(self, disparity_table, x_metric, y_metric='precision',
                              x_agg_method='mean', y_agg_method='mean', title=True,
                              x_jitter=None, y_jitter=None, selected_models=None,
                              ax=None, scatter_kws={'legend': 'full'}, show_figure=True):
        """
        Compare two absolute bias metrics or bias metric disparities across models.

        :param disparity_table: disparity table. output of bias.get_disparity, or
            fairness.get_fairness function.
        :param x_metric: the metric to plot on the X axis. Must be a column in the disparity_table.
        :param y_metric: the metric to plot on the Y axis. Must be a column in the disparity_table.
        :param x_agg_method: Method to aggregate metric values for X axis. Options: 'mean', 'median', 'max', 'min'. Default is 'mean'. For absolute metrics, 'mean' aggregation is a weighted average by group size.
        :param y_agg_method: Method to aggregate metric values for Y axis. Options: 'mean', 'median', 'max', 'min'. Default is 'mean'. For absolute metrics, 'mean' aggregation is a weighted average by group size.
        :param title: whether to include a title in visualizations. Default is True.
        :param x_jitter: jitter for x values. Default is None.
        :param y_jitter: jitter for y values. Default is None.
        :param selected_models: which models to visualize. Default is all models in disparity_table.
        :param ax: a matplotlib Axis. If not passed, a new figure will be created.
        :param scatter_kws: keyword arguments for scatterplot
        :param show_figure: whether to show figure (plt.show()). Default is True.

        :return: A Matplotlib axis
        """

        df_models = self._check_multiple_models(disparity_table, method_table_name='disparities_table')

        if not selected_models:
            selected_models = df_models

        plot_table = disparity_table.loc[disparity_table['model_id'].isin(selected_models)]

        # requirement: at least two model_id values
        if len(selected_models) < 2:
            raise ValueError("This method requires at least two distinct 'model_id' values "
                             "in the disparities table. Tip: check that "
                             "disparities_table.model_id.unique() returns more than one element.")

        # must be valid metric
        if x_metric not in plot_table.columns:
            raise ValueError(
                f"Absolute metric '{x_metric}' is not included in disparities_table.")

        if y_metric not in plot_table.columns:
            raise ValueError(
                f"Disparity metric '{y_metric}' is not included in disparities_table.")

        # must be valid aggregation method
        if (x_agg_method not in ('mean', 'median', 'max', 'min')) or (y_agg_method not in ('mean', 'median', 'max', 'min')):
            raise ValueError(
                "Aggregation methods 'x_agg_method' and 'y_agg_method' must "
                "take one of the following values: 'mean', 'median', 'max', 'min'.")


        # should never really have NaNs for one model but not another, but handling JIC
        get_indices = lambda x: ~np.isnan(x)
        get_weights = lambda x: plot_table.loc[x.index, "group_size"]
        wtd_mean = lambda x: (np.average(x[get_indices(x)], axis=0, weights=get_weights(x)[get_indices(x)]))

        if x_agg_method == "mean":
            if "_disparity" not in x_metric:
                x_agg_method = wtd_mean

        if y_agg_method == "mean":
            if "_disparity" not in y_metric:
                y_agg_method = wtd_mean

        collected_df = plot_table.groupby('model_id', as_index=False).agg({x_metric: x_agg_method, y_metric:y_agg_method})

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        aq_palette = sns.diverging_palette(225, 35, sep=10, n=40, as_cmap=True, center="dark")

        with sns.axes_style("whitegrid"):

            ax = sns.scatterplot(x=x_metric, y=y_metric, data=collected_df, hue='model_id',
                                 x_jitter=x_jitter, y_jitter=y_jitter, palette=aq_palette,
                                 alpha=0.75, **scatter_kws)

        ax.xaxis.grid(color='lightgray', which='major')
        ax.yaxis.grid(color='lightgray', which='major')
        labels = [item.get_text().replace('_', ' ').upper() for item in ax.get_xticklabels()]
        if '' not in labels:
            ax.set_xticklabels(labels, rotation=30, ha='center')
        else:
            plt.xticks(rotation=30, horizontalalignment='center')

        x_clean = x_metric.replace('_', ' ').upper()
        y_clean = y_metric.replace('_', ' ').upper()
        ax.set_xlabel(x_clean, fontsize=12)
        ax.set_ylabel(y_clean, fontsize=12)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=[f"Model {model}" for model in labels[1:]], title="Model ID")

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if title:
            plot_title = f"MODEL COMPARISON: {y_clean} BY {x_clean}"
            ax.set_title(plot_title, fontsize=20)

        if show_figure:
            plt.show()

        else:
            return ax
