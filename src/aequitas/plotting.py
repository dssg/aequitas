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
        list(disparities_table.columns[disparities_table.columns.str.contains(
            ref_group_flag)])

    if specific_measures:
        ref_group_cols = \
            [measure + ref_group_flag for measure in specific_measures if
             measure + ref_group_flag in ref_group_cols]

    attributes = list(disparities_table.attribute_name.unique())
    for attribute in attributes:
        attr_table = \
            disparities_table.loc[disparities_table['attribute_name'] == attribute]
        attr_refs = {}
        for col in ref_group_cols:
            metric_key = "".join(col.split(ref_group_flag))
            attr_refs[metric_key] = \
                attr_table.loc[attr_table['attribute_name'] == attribute, col].min()
        if label_score_ref:
            if label_score_ref + ref_group_flag in ref_group_cols:
                attr_refs['label_value'] = attr_refs[label_score_ref]
                attr_refs['score'] = attr_refs[label_score_ref]
            else:
                raise ValueError("The specified reference measure for label"
                                 " value and score is not included in the "
                                 "data frame.")

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
                                 ref_group_flag='_ref_group_value', model_id=1):
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
        :param model_id: model ID number. Default is 1.

        :return: Integer indicating relative index of reference group value row.
        """
        # get absolute metric name from passed group metric (vs. a disparity name)
        abs_metric = "".join(group_metric.split('_disparity'))

        all_ref_groups = assemble_ref_groups(disparities_table, ref_group_flag)
        ref_group_name = all_ref_groups[attribute_name][abs_metric]

        # get index for row associated with reference group for that model
        ind = list(disparities_table.loc[(disparities_table['attribute_name'] == attribute_name) &
                                         (disparities_table['attribute_value'] == ref_group_name) &
                                         (disparities_table['model_id'] == model_id)].index)

        # there should only ever be one item in list, but JIC, select first
        idx = ind[0]

        relative_ind = disparities_table.index.get_loc(idx)
        return relative_ind, ref_group_name

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
        if group_metric not in group_table.columns:
            raise ValueError(f"Specified disparity metric '{group_metric}' not "
                             f"in 'group_table'.")

        if group_table[group_metric].isnull().any():
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
                group_label = f"{label} ({g_size:,})"

                if ax_lim < 3:
                    CHAR_PLACEHOLDER = 0.03
                else:
                    CHAR_PLACEHOLDER = 0.25

                label_length = len(label) * CHAR_PLACEHOLDER
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
            ax.set_title(f"{group_metric.upper()}", fontsize=20)

        return ax

    def plot_disparity(self, disparity_table, group_metric, attribute_name,
                       color_mapping=None, model_id=1, ax=None, fig=None,
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
        :param model_id: model ID number. Default is 1.
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
        if sorted_df.columns[
            sorted_df.columns.str.contains('_significance')].value_counts().sum() > 0:
        # unmasked significance
        # find indices where related significance have smaller value than significance_alpha
            if np.issubdtype(
                    sorted_df[
                        self._significance_disparity_mapping[related_disparity]].dtype,
                    np.number):
                to_star = sorted_df.loc[
                    sorted_df[
                        self._significance_disparity_mapping[related_disparity]] < significance_alpha].index.tolist()


            # masked significance
            # find indices where attr values have True value for each of those two columns,
            else:
                to_star = sorted_df.loc[
                    sorted_df[
                        self._significance_disparity_mapping[related_disparity]] > 0].index.tolist()


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
        if not ax or not fig:
            fig, ax = plt.subplots(figsize=(5, 4))

        ax = sf.squarify_plot_rects(padded_rects, color=clrs, labels=labels,
                                 values=label_values, ax=ax, alpha=0.8,
                                    acronyms=False)

        # TO DO: build out in next phase (model comparison)
        # if model_id:
        #     ax.set_title(f"MODEL {model_id}, {(' ').join(group_metric.split('_')).upper()} ({attribute_name.upper()})",
        #              fontsize=23, fontweight="bold")

        if title:
            ax.set_title(f"{(' ').join(related_disparity.split('_')).upper()} ({attribute_name.upper()})",
                     fontsize=23, fontweight="bold")

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

        :param fairness_table: a fairness table. Output of fairness.get_fairness
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

                if ax_lim < 3:
                    CHAR_PLACEHOLDER = 0.03
                else:
                    CHAR_PLACEHOLDER = 0.25

                label_length = len(label) * CHAR_PLACEHOLDER
                max_val_length = 7 * CHAR_PLACEHOLDER
                indent_length = ax_lim * 0.025

                # bar long enough for label, enough space after bar for value
                if ((indent_length + label_length) < (next_position - indent_length)) and (
                        (next_position + indent_length + max_val_length) < (
                        ax_lim - indent_length)):

                    ax.text(next_position + indent_length, y + float(height_of_bar) / 2,
                            f"{value:.2f}", fontsize=12, verticalalignment='top')
                    ax.text(indent_length, y + float(height_of_bar) / 2,
                            label, fontsize=11, verticalalignment='top',
                            color=text_color)

                # case when bar too long for labels after bar, print all text in bar
                elif (next_position + indent_length + max_val_length) > (
                        ax_lim - indent_length):

                    ax.text(indent_length, y + float(height_of_bar) / 2,
                            f"{label}, {value:.2f}", fontsize=11,
                            verticalalignment='top', color=text_color)

                # case when bar too small for labels inside bar, print all text
                # after bar
                else:
                    ax.text(next_position + indent_length,
                            y + float(height_of_bar) / 2,
                            f"{label}, {value:.2f}", fontsize=12,
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
                                attribute_name, model_id=1, ax=None, fig=None,
                                title=True, min_group_size=None,
                                significance_alpha=0.05):
        """
        Plot disparity metrics colored based on calculated disparity.

        :param group_metric: the metric to plot. Must be a column in the disparity_table.
        :param attribute_name: which attribute to plot group_metric across.
        :param model_id: model ID number. Default is 1.
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
                                   color_mapping=None, model_id=model_id,
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
                f"DISPARITY METRICS by {(', ').join(list(map(lambda x:x.upper(), attributes)))}"

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
        if num_metrics > 1:
            ncols = 3
        else:
            ncols = 1

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

        models = list(data_table.model_id.unique())

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
                           fillzeros=True, title=True, label_dict=None, ncols=3,
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

        :param fairness_table: output of fairness.get_fairness functions.
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

        :param fairness_table: output of fairness.get_fairness functions.
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
