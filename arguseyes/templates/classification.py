from mlinspect import PipelineInspector
from mlinspect.inspections import RowLineage

from arguseyes.extractors import source_extractor, feature_matrix_extractor


class ClassificationPipeline:

    def __init__(self, result, lineage_inspection, train_sources, X_train, X_test, y_train, y_test):
        self.result = result
        self.lineage_inspection = lineage_inspection
        self.train_sources = train_sources
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def _from_result(result, lineage_inspection):

        train_sources = source_extractor.extract_train_sources(result, lineage_inspection)

        X_train = feature_matrix_extractor.extract_train_feature_matrix(result, lineage_inspection)
        X_test = feature_matrix_extractor.extract_test_feature_matrix(result, lineage_inspection)

        y_train = feature_matrix_extractor.extract_train_labels(result, lineage_inspection)
        y_test = feature_matrix_extractor.extract_test_labels(result, lineage_inspection)

        return ClassificationPipeline(result, lineage_inspection, train_sources, X_train, X_test, y_train, y_test)

    @staticmethod
    def from_py_file(path_to_py_file):
        # TODO we need to get rid of this
        num_records = 1000000
        lineage_inspection = RowLineage(num_records)

        result = PipelineInspector \
            .on_pipeline_from_py_file(path_to_py_file) \
            .add_required_inspection(lineage_inspection) \
            .execute()

        return ClassificationPipeline._from_result(result, lineage_inspection)


    @staticmethod
    def from_notebook(path_to_ipynb_file):
        # TODO we need to get rid of this
        num_records = 100000
        lineage_inspection = RowLineage(num_records)

        result = PipelineInspector \
            .on_pipeline_from_ipynb_file(path_to_ipynb_file) \
            .add_required_inspection(lineage_inspection) \
            .execute()

        return ClassificationPipeline._from_result(result, lineage_inspection)