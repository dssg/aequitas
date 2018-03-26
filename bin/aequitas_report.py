import argparse
import logging
from sys import exit

import yaml

from bin.utils.db import get_engine
from bin.utils.report import audit_report

# Authors: Pedro Saleiro <saleiro@uchicago.edu>
#          Rayid Ghani
#
# License: Copyright \xa9 2018. The University of Chicago. All Rights Reserved.

about = """
############################################################################
##   Center for Data Science and Public Policy                            ##
##   http://dsapp.uchicago.edu                                            ##
##                                                                        ##
##   Copyright \xa9 2018. The University of Chicago. All Rights Reserved.    ## 
############################################################################



____________________________________________________________________________
                                     _ _            
               __ _  ___  __ _ _   _(_) |_ __ _ ___ 
              / _` |/ _ \/ _` | | | | | __/ _` / __| 
             | (_| |  __/ (_| | |_| | | || (_| \__ \ 
              \__,_|\___|\__, |\__,_|_|\__\__,_|___/
                            |_|                            

____________________________________________________________________________  

                    Bias and Fairness Analysis


"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=about + 'Creates a pdf report based on the input group_value_fairness_report df\n')

    parser.add_argument('--input',
                        action='store',
                        dest='input_file',
                        default=None,
                        help='Absolute filepath for input group_value_fairness_report data frame in csv format. If no input is provided we '
                             'assume there is a '
                             'db configuration in the configs.yaml file.')

    parser.add_argument('--config',
                        action='store',
                        dest='config_file',
                        default='configs/configs.yaml',
                        help='Absolute filepath for input yaml config file. Default is configs/configs.yaml')

    parser.add_argument('--output-folder',
                        action='store',
                        dest='output_folder',
                        default='output/',
                        help='Folder name to be created inside aequitas/output/')

    return parser.parse_args()


def main():
    args = parse_args()
    print(about)
    configs = None
    try:
        with open(args.config_file) as f:
            configs = yaml.load(f)
    except FileNotFoundError:
        logging.error('Could not load configurations! Please set configs.yaml file using --config')
        exit()
    if configs is None:
        logging.error('Empty configurations! Please set configs.yaml file using --config')
        exit()

    # when having score vs prediction compatibility, thresholds only make sense for score
    if args.input_file is None:
        engine = get_engine(configs)

    audit_report(model_id, parameter, attributes, model_eval, configs, fair_results,
                 f.fair_measures, ref_groups_method, group_value_report)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
