from scipy import stats

from arguseyes.issues._issue import IssueDetector, Issue


class LabelShift(IssueDetector):

    def _detect(self, pipeline) -> Issue:
        # TODO this assumes binary classification at the moment, needs to be generalised
        num_pos_train = pipeline.y_train.sum()
        num_neg_train = len(pipeline.y_train) - num_pos_train

        num_pos_test = pipeline.y_test.sum()
        num_neg_test = len(pipeline.y_test) - num_pos_test

        threshold = 0.01
        _, p_value, _, _ = stats.chi2_contingency([[num_pos_train, num_neg_train], [num_pos_test, num_neg_test]])
        label_shift = p_value < threshold

        return Issue('label_shift', label_shift, {'threshold': threshold, 'p_value': p_value})
