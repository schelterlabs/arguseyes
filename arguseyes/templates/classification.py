import os
import mlflow
import logging
import json

from arguseyes.issues import IssueDetector
from arguseyes.refinements import Refinement
from arguseyes.templates import Output



class ClassificationPipeline:

    def __init__(self, train_sources, train_source_lineage, test_sources, test_source_lineage,
                 outputs, output_lineage):

        self.train_sources = train_sources
        self.train_source_lineage = train_source_lineage
        self.test_sources = test_sources
        self.test_source_lineage = test_source_lineage
        self.outputs = outputs
        self.output_lineage = output_lineage

        self._log_pipeline_details()


    def _log_pipeline_details(self):

        X_train = self.outputs[Output.X_TRAIN]
        X_test = self.outputs[Output.X_TEST]

        for train_source in self.train_sources:
            source_id = train_source.operator_id
            mlflow.log_param(f'arguseyes.train_source.{source_id}.type', train_source.source_type)
            mlflow.log_param(f'arguseyes.train_source.{source_id}.num_records', len(train_source.data))            
            mlflow.log_param(f'arguseyes.train_source.{source_id}.attributes', list(train_source.data.columns)) 
            mlflow.log_param(f'arguseyes.train_source.{source_id}.attribute_types', 
                [str(dtype) for dtype in train_source.data.dtypes])    

        for test_source in self.test_sources:
            source_id = test_source.operator_id
            mlflow.log_param(f'arguseyes.test_source.{source_id}.type', test_source.source_type)
            mlflow.log_param(f'arguseyes.test_source.{source_id}.num_records', len(test_source.data))
            mlflow.log_param(f'arguseyes.test_source.{source_id}.attributes', list(test_source.data.columns))
            mlflow.log_param(f'arguseyes.test_source.{source_id}.attribute_types',
                [str(dtype) for dtype in test_source.data.dtypes])

        mlflow.log_param("arguseyes.X_train.num_rows", X_train.shape[0])
        mlflow.log_param("arguseyes.X_train.num_features", X_train.shape[1])
        mlflow.log_param("arguseyes.X_test.num_rows", X_test.shape[0])
        mlflow.log_param("arguseyes.X_test.num_features", X_test.shape[1])




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
        return refinement._compute(self)
