import click
import os
import sys
import yaml
import logging
import mlflow

from arguseyes import ArgusEyes
from arguseyes.issues import *


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@click.command()
@click.argument("yaml_file")
def main(yaml_file):

    logging.info('')
    logging.info(f'#' * 80)
    logging.info('  SETTING UP ARGUS EYES')
    logging.info(f'#' * 80)

    logging.info(f'Reading configuration from {yaml_file}...')
    with open(yaml_file, 'r') as stream:
        config = yaml.safe_load(stream)

    pipeline_config = config['pipeline']
    issues_to_detect = pipeline_config['detect_issues']

    synthetic_cmd_args = []

    if 'args' in pipeline_config:
        synthetic_cmd_args = pipeline_config['args']
        logging.info(f"Synthetic commandline arguments specified: {synthetic_cmd_args}")

    series = config['series']
    mlflow_artifact_storage_uri = os.path.expandvars(config['artifact_storage_uri'])
    pipeline_path = os.path.expandvars(pipeline_config['path'])

    working_directory = os.path.dirname(pipeline_path)
    logging.info(f'Changing to directory {working_directory}...')
    os.chdir(working_directory)

    issue_detectors_by_name = {
        'constant_features': ConstantFeatures(),
        'unnormalised_features': UnnormalisedFeatures(),
        'label_shift': LabelShift(),
        'covariate_shift': CovariateShift(),
        'data_leakage': DataLeakage(),
        'fairness': Fairness(),
        'label_errors': LabelErrors(),
    }

    logging.info(f'Storing artifacts via mlflow at {mlflow_artifact_storage_uri}...')

    logging.info('')
    logging.info(f'#' * 80)
    logging.info('  STARTING INSTRUMENTED EXECUTION OF THE ML PIPELINE')
    logging.info(f'#' * 80)
    logging.info(f'Executing pipeline {pipeline_path} for the series {series}')
    eyes = ArgusEyes(series, mlflow_artifact_storage_uri)


    pipeline = eyes.classification_pipeline_from_py_file(pipeline_path, cmd_args=synthetic_cmd_args)

    logging.info('')
    logging.info(f'#' * 80)
    logging.info('  SCREENING PIPELINE FOR ISSUES')
    logging.info(f'#' * 80)

    issue_detected = False

    for an_issue in issues_to_detect:

        issue = an_issue['issue']
        issue_name = issue['name']
        issue_params = {}
        if 'params' in issue:
            issue_params = issue['params']

        issue_detector = issue_detectors_by_name[issue_name]
        logging.info(f'Looking for issue {issue_name}...')
        issue = pipeline.detect_issue(issue_detector, issue_params)
        if not issue.is_present:
            logging.info('Not found.')
        else:
            logging.warning(
                '\x1b[31;21m' +
                '\n\n' +
                '-' * 80 +
                f'\n{issue_detector.error_msg(issue)}\n' +
                '-' * 80 + '\n\x1b[0m')
            issue_detected = True

    mlflow_run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    if issue_detected:
        logging.error(
            '\x1b[31;21m' +
            '\n\n' +
            '-' * 80 +
            '\n Pipeline fails ArgusEyes screening!\n' +
            f' Check the mlflow run {mlflow_run_id} for details.\n' +
            '-' * 80 + '\n\x1b[0m')
        sys.exit(os.EX_DATAERR)
    else:
        logging.info(
            '\x1b[33;92m' +
            '\n\n' +
            '-' * 80 +
            '\n Pipeline passes ArgusEyes screening\n' +
            '-' * 80 + '\n\x1b[0m')

        sys.exit(os.EX_OK)


if __name__ == "__main__":
    main()
