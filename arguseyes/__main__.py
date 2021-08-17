import click
import os
import sys
import yaml
import logging
import mlflow

from arguseyes import ArgusEyes
from arguseyes.issues import *
from arguseyes.refinements import *


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@click.command()
@click.argument("yaml_file")
def main(yaml_file):
    logging.info(f'Reading configuration from {yaml_file}...')
    with open(yaml_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # TODO sanity checking of yaml contents
    # TODO we should have constants for all the string keys here

    pipeline_config = config['pipeline']
    issues_to_detect = pipeline_config['detect_issues']

    working_directory = pipeline_config['working_directory']
    logging.info(f'Changing to directory {working_directory}...')
    os.chdir(working_directory)

    synthetic_cmd_args = []

    if 'args' in pipeline_config:
        synthetic_cmd_args = pipeline_config['args']
        logging.info(f"Synthetic commandline arguments specified: {synthetic_cmd_args}")

    series = config['series']
    mlflow_artifact_storage_uri = config['artifact_storage_uri']
    pipeline_path = pipeline_config['path']

    issues_by_name = {
        'constant_features': ConstantFeatures(),
        'unnormalised_features': UnnormalisedFeatures(),
        'label_shift': LabelShift(),
        'covariate_shift': CovariateShift(),
        'traintest_overlap': TrainTestOverlap()
    }

    logging.info(f'Storing artifacts via mlflow at {mlflow_artifact_storage_uri}...')
    logging.info(f'Executing pipeline {pipeline_path} for the series {series}')
    eyes = ArgusEyes(series, mlflow_artifact_storage_uri)


    with eyes.classification_pipeline_from_py_file(pipeline_path, cmd_args=synthetic_cmd_args) as pipeline:

        issue_detected = False

        for issue_name, issue_detector in issues_by_name.items():
            if issues_to_detect is not None and issue_name in issues_to_detect:
                logging.info(f'Looking for issue {issue_name}...')
                issue = pipeline.detect_issue(issue_detector)
                if not issue.is_present:
                    logging.info('Not found.')
                else:
                    logging.warning(
                        '\x1b[31;21m' + \
                        '\n\n' + \
                        '-' * 80 + \
                        f'\n{issue_detector.error_msg(issue)}\n' + \
                        '-' * 80 + '\n\x1b[0m')
                    issue_detected = True

        if 'data-refinements' in pipeline_config:
            for refinement in pipeline_config['data-refinements']:
                refinement_info = refinement['refinement']
                if refinement_info['name'] == 'input_usage':
                    logging.info('Computing usage information for input records')
                    pipeline.compute(InputUsage())
                if refinement_info['name'] == 'data_valuation':
                    if 'params' in refinement_info:
                        k = refinement_info['params']['k']
                        num_test_samples = refinement_info['params']['num_test_samples']
                        logging.info(f'Computing data valuation information for input records with k={k}' +
                                    f' and {num_test_samples} test samples')
                        pipeline.compute(DataValuation(k, num_test_samples))
                    else:
                        logging.info('Computing data valuation information for input records')
                        pipeline.compute(DataValuation())
                if refinement_info['name'] == 'fairness_metrics':
                    sensitive_attribute = refinement_info['params']['sensitive_attribute']
                    non_protected_class = refinement_info['params']['non_protected_class']
                    logging.info(f'Computing fairness metrics with {sensitive_attribute} as sensitive attribute' +
                                f' and {non_protected_class} as non-protected class')
                    pipeline.compute(FairnessMetrics(sensitive_attribute, non_protected_class))

    if issue_detected:
        logging.error(
            '\x1b[31;21m' + \
            '\n\n' + \
            '-' * 80 + \
            '\n Pipeline fails ArgusEyes screening\n' + \
            '-' * 80 + '\n\x1b[0m')
        sys.exit(os.EX_DATAERR)
    else:
        logging.info(
            '\x1b[33;92m' + \
            '\n\n' + \
            '-' * 80 + \
            '\n Pipeline passes ArgusEyes screening\n' + \
            '-' * 80 + '\n\x1b[0m')

        sys.exit(os.EX_OK)


if __name__ == "__main__":
    main()
