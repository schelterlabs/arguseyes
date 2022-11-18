import numpy as np
from numba import njit, prange
import logging

from arguseyes.issues import Issue, IssueDetector
from arguseyes.templates import SourceType, Output


# removed cache=True because of https://github.com/numba/numba/issues/4908 need a workaround soon
@njit(fastmath=True, parallel=True)
def _compute_shapley_values(X_train, y_train, X_test, y_test, K=1):
    N = len(X_train)
    M = len(X_test)
    result = np.zeros(N, dtype=np.float32)

    for j in prange(M):
        score = np.zeros(N, dtype=np.float32)
        dist = np.zeros(N, dtype=np.float32)
        div_range = np.arange(1.0, N)
        div_min = np.minimum(div_range, K)
        for i in range(N):
            dist[i] = np.sqrt(np.sum(np.square(X_train[i] - X_test[j])))
        indices = np.argsort(dist)
        y_sorted = y_train[indices]
        eq_check = (y_sorted == y_test[j]) * 1.0
        diff = - 1 / K * (eq_check[1:] - eq_check[:-1])
        diff /= div_range
        diff *= div_min
        score[indices[:-1]] = diff
        score[indices[-1]] = eq_check[-1] / N
        score[indices] += np.sum(score[indices]) - np.cumsum(score[indices])
        result += score / M

    return result


class LabelErrors(IssueDetector):

    def detect(self, pipeline, params) -> Issue:

        if 'k' in params:
            k = params['k']
        else:
            k = 100

        threshold = params['max_fraction']

        X_train = pipeline.outputs[Output.X_TRAIN]
        y_train = pipeline.outputs[Output.Y_TRAIN]

        X_test = pipeline.outputs[Output.X_TEST]
        y_test = pipeline.outputs[Output.Y_TEST]

        # Still hacky, we need a principled way to flatten tensors for CV pipelines
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(int(X_train.shape[0] / X_train.shape[1]), X_train.shape[1] * X_train.shape[1])
            X_test = X_test.reshape(int(X_test.shape[0] / X_test.shape[1]), X_test.shape[1] * X_test.shape[1])

        shapley_values = _compute_shapley_values(X_train,
                                                 np.squeeze(y_train),
                                                 X_test,
                                                 np.squeeze(y_test),
                                                 k)

        lineage_X_train = pipeline.output_lineage[Output.X_TRAIN]

        fact_table_index, fact_table_source = \
            [(index, train_source) for index, train_source in enumerate(pipeline.train_sources)
             if train_source.source_type == SourceType.ENTITIES][0]

        shapley_values_by_row_id = {}

        for polynomial, shapley_value in zip(lineage_X_train, shapley_values):
            for entry in polynomial:
                if entry.operator_id == fact_table_source.operator_id:
                    shapley_values_by_row_id[entry.row_id] = shapley_value

        data = fact_table_source.data
        fact_table_lineage = pipeline.train_source_lineage[fact_table_index]

        for row_index, row in data.iterrows():                
            data.at[row_index, '__shapley_value'] = \
                self._find_shapley(fact_table_lineage[row_index], shapley_values_by_row_id)

        self.log_tag('arguseyes.shapley_values.operator_id', fact_table_source.operator_id)
        self.log_tag('arguseyes.shapley_values.k', k)
        self.log_tag('arguseyes.shapley_values.data_file', 'input-with-shapley-values.parquet')
        self.log_as_parquet_file(data, 'input-with-shapley-values.parquet')


        num_samples = len(shapley_values_by_row_id)
        num_negative = int(np.sum(np.array(list(shapley_values_by_row_id.values())) < 0, axis=0))
        fraction_negative = float(num_negative) / num_samples

        logging.info(f'Found {num_negative} out of {num_samples} samples with negative Shapley value.')



        has_too_many_label_errors = fraction_negative > threshold

        issue_details = {
            'num_samples': num_samples,
            'num_erroneous': num_negative,
            'fraction': fraction_negative
        }

        return Issue('label_errors', has_too_many_label_errors, issue_details)



    def error_msg(self, issue) -> str:
        details = issue.details
        return f'Found {details["fraction"]*100:.2f}% ({details["num_erroneous"]}/{details["num_samples"]}) ' \
               'of potentially mislabeled samples in the training data!'



    @staticmethod
    def _find_shapley(polynomial, shapley_values_by_row_id):
        for entry in polynomial:
            if entry.row_id in shapley_values_by_row_id:
                return shapley_values_by_row_id[entry.row_id]
        return 0.0
