import os
import mlflow
from PIL import Image
import json
import tempfile
from contextlib import redirect_stdout

from mlinspect import PipelineInspector
from mlinspect.inspections import RowLineage
from mlinspect.visualisation import save_fig_to_path
from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.issues._issue import IssueDetector
from arguseyes.refinements._refinement import Refinement
from arguseyes.templates.extractors import feature_matrix_extractor
from arguseyes.templates.extractors import source_extractor


class ClassificationPipeline:

    def __init__(self, result, lineage_inspection, train_sources, test_sources, X_train, X_test, y_train, y_test):
        self.result = result
        self.lineage_inspection = lineage_inspection
        self.train_sources = train_sources
        self.test_sources = test_sources
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self._log_mlinspect_results()
        self._log_pipeline_details()

    def _log_pipeline_details(self):

        mlflow.log_param("arguseyes.X_train.num_rows", self.X_train.shape[0])
        mlflow.log_param("arguseyes.X_train.num_features", self.X_train.shape[1])
        mlflow.log_param("arguseyes.X_test.num_rows", self.X_test.shape[0])
        mlflow.log_param("arguseyes.X_test.num_features", self.X_test.shape[1])

        with tempfile.TemporaryDirectory() as tmpdirname:
            dag_filename = os.path.join(tmpdirname, 'arguseyes-dag.png')
            save_fig_to_path(self.result.dag, dag_filename)
            dag_image = Image.open(dag_filename).convert("RGB")
            mlflow.log_image(dag_image, 'arguseyes-dag.png')

    def _log_mlinspect_results(self):
        # TODO @Shubha this is where we should serialise the DAG to json and log it as a tag to mlflow
        # TODO @Shubha this is also where should log the intermediate results from the lineage inspection as artifacts
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        pass

    def detect_issue(self, issue_detector: IssueDetector):
        issue = issue_detector._detect(self)

        mlflow.set_tag(f'arguseyes.issues.{issue.id}.is_present', issue.is_present)
        mlflow.set_tag(f'arguseyes.issues.{issue.id}.details', json.dumps(issue.details))

        return issue

    def compute(self, refinement: Refinement):
        # TODO Not sure whether it makes sense to persist these potentially large outputs
        return refinement._compute(self)

    @staticmethod
    def _from_result(result, lineage_inspection):

        # TODO persist with mlflow

        train_sources = source_extractor.extract_train_sources(result, lineage_inspection)
        test_sources = source_extractor.extract_test_sources(result, lineage_inspection)

        X_train = feature_matrix_extractor.extract_train_feature_matrix(result, lineage_inspection)
        X_test = feature_matrix_extractor.extract_test_feature_matrix(result, lineage_inspection)

        y_train = feature_matrix_extractor.extract_train_labels(result, lineage_inspection)
        y_test = feature_matrix_extractor.extract_test_labels(result, lineage_inspection)

        return ClassificationPipeline(result, lineage_inspection, train_sources, test_sources,
                                      X_train, X_test, y_train, y_test)

    @staticmethod
    def _execute_pipeline(inspector: PipelineInspector):
        lineage_inspection = RowLineage(RowLineage.ALL_ROWS, [OperatorType.DATA_SOURCE, OperatorType.TRAIN_DATA,
                                                              OperatorType.TRAIN_LABELS, OperatorType.TEST_DATA,
                                                              OperatorType.TEST_LABELS, OperatorType.SCORE,
                                                              OperatorType.JOIN])
        mlflow.start_run()

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, 'arguseyes-pipeline-output.txt'), 'w') as tmpfile:
                with redirect_stdout(tmpfile):
                    result = inspector \
                        .add_required_inspection(lineage_inspection) \
                        .execute()
                    mlflow.log_artifact(tmpfile.name)

        return ClassificationPipeline._from_result(result, lineage_inspection)

    @staticmethod
    def from_py_file(path_to_py_file, cmd_args=[]):
        synthetic_cmd_args = ['eyes']
        synthetic_cmd_args.extend(cmd_args)
        from unittest.mock import patch
        import sys
        with patch.object(sys, 'argv', synthetic_cmd_args):
            return ClassificationPipeline._execute_pipeline(PipelineInspector.on_pipeline_from_py_file(path_to_py_file))

    @staticmethod
    def from_notebook(path_to_ipynb_file):
        return ClassificationPipeline._execute_pipeline(
            PipelineInspector.on_pipeline_from_ipynb_file(path_to_ipynb_file))
