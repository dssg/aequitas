import logging
import os

import yaml

logging.getLogger(__name__)


class Configs(object):
    def __init__(self, ref_groups_method='majority', fairness_threshold=0.8,
                 attr_cols=None, report=True, score_thresholds=None, project_description={'title': '', 'goal': ''},
                 ref_groups=None, db=None, fairness_measures=['Statistical Parity', 'Impact Parity', 'FDR Parity',
                                                              'FPR Parity', 'FNR Parity', 'FOR Parity']):
        self.ref_groups_method = ref_groups_method
        self.fairness_threshold = fairness_threshold
        self.attr_cols = attr_cols
        self.report = report
        self.score_thresholds = score_thresholds
        self.ref_groups = ref_groups
        self.db = db
        self.fair_measures_requested = fairness_measures
        self.project_description = project_description

    @staticmethod
    def load_configs(configs_path='aequitas_cli/configs/config.yaml'):
        try:

            with open(configs_path, 'r') as stream:
                configs_fromfile = yaml.load(stream)
        except FileNotFoundError:
            logging.error('**Could not find configurations! Please set path to configs.yaml file using --config')
            print(configs_path)
            print(os.getcwd())
            exit(1)
        return Configs(**configs_fromfile)
