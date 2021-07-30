import numpy as np

from arguseyes.issues import IssueDetector, Issue
from arguseyes.templates import Output

class ConstantFeatures(IssueDetector):

    @staticmethod
    def _is_constant(column):
        unique_values = np.unique(column)
        return len(unique_values) == 1

    def error_msg(self, issue) -> str:
        num_constant_features = len(issue.details['constant_feature_indices'])
        return f'Encountered {num_constant_features} constant feature(s)!'

    def _detect(self, pipeline) -> Issue:

        #from pprint import pprint   
        #pprint(vars(pipeline))

        X_train = pipeline.outputs[Output.X_TRAIN]
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
