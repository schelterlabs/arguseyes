import numpy as np

from arguseyes.issues._issue import IssueDetector, Issue


class ConstantFeatures(IssueDetector):

    @staticmethod
    def _is_constant(column):
        unique_values = np.unique(column)
        return len(unique_values) == 1

    def _detect(self, pipeline) -> Issue:

        X_train = pipeline.X_train
        _, num_columns = X_train.shape

        constant_column_found = False

        issue_details = {
            'constant_feature_indices': [],
            'nonconstant_feature_indices': []
        }

        for column_index in range(0, num_columns):
            column = X_train[:, column_index]
            if self._is_constant(column):
                issue_details['constant_feature_indices'].append(column_index)
                constant_column_found = True
            else:
                issue_details['nonconstant_feature_indices'].append(column_index)

        return Issue('constant_features', constant_column_found, issue_details)
