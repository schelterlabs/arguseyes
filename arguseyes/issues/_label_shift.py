from scipy import stats

from arguseyes.issues import IssueDetector, Issue
from arguseyes.templates import Output

class LabelShift(IssueDetector):

    def _detect(self, pipeline) -> Issue:

        y_train = pipeline.outputs[Output.Y_TRAIN]    
        y_test = pipeline.outputs[Output.Y_TEST]

        # TODO this assumes binary classification at the moment, needs to be generalised
        num_pos_train = y_train.sum()
        num_neg_train = len(y_train) - num_pos_train

        num_pos_test = y_test.sum()
        num_neg_test = len(y_test) - num_pos_test

        threshold = 0.001
        _, p_value, _, _ = stats.chi2_contingency([[num_pos_train, num_neg_train], [num_pos_test, num_neg_test]])
        label_shift = p_value < threshold

        return Issue('label_shift', label_shift, {'threshold': threshold, 'p_value': p_value})

    def error_msg(self, issue) -> str:
        return f'Found {issue.id}, independence test between label frequencies in train and\n' + \
               f'test set failed with a p-value of {issue.details["p_value"]} < {issue.details["threshold"]}!'