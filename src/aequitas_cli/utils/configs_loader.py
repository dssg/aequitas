import logging
import os

import yaml

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."

class Configs(object):
    original_fairness_measures = (
        'Statistical Parity', 'Impact Parity', 'FDR Parity',
        'FPR Parity', 'FNR Parity', 'FOR Parity', 'TPR Parity',
        'Precision Parity')

    def __init__(self, ref_groups_method='majority', fairness_threshold=0.8,
                 attr_cols=None, report=True, score_thresholds=None,
                 project_description={'title': '', 'goal': ''},
                 ref_groups=None, db=None,
                 fairness_measures=original_fairness_measures,
                 plot_bias_disparities=(), plot_bias_metrics=()):

        self.ref_groups_method = ref_groups_method
        self.fairness_threshold = fairness_threshold
        self.attr_cols = attr_cols
        self.report = report
        self.score_thresholds = score_thresholds
        self.ref_groups = ref_groups
        self.db = db
        self.fair_measures_requested = list(fairness_measures)
        self.project_description = project_description
        self.plot_bias_metrics = plot_bias_metrics
        self.plot_bias_disparities = plot_bias_disparities

    @staticmethod
    def load_configs(configs_path):
        try:
            if configs_path:
                with open(configs_path, 'r') as stream:
                    configs_fromfile = yaml.load(stream)
            else:
                configs_fromfile = {}
        except FileNotFoundError:
            logging.error('**Could not find configurations! Please set path to configs.yaml file using --config')
            print(configs_path)
            print(os.getcwd())
            exit(1)
        return Configs(**configs_fromfile)
