import numpy as np
from numba import njit, prange

from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.refinements._refinement import Refinement
from arguseyes.templates.source import SourceType, Source
from arguseyes.utils.dag_extraction import find_dag_node_by_type


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


class DataValuation(Refinement):

    def __init__(self, k=1, num_test_samples=10):
        self.k = k
        self.num_test_samples = num_test_samples

    def _compute(self, pipeline):
        result = pipeline.result
        lineage_inspection = pipeline.lineage_inspection

        X_train = pipeline.X_train
        X_test = pipeline.X_test
        y_train = pipeline.y_train
        y_test = pipeline.y_test

        X_test_sampled = X_test[:self.num_test_samples, :]
        y_test_sampled = y_test[:self.num_test_samples, :]

        print("X_train", X_train.shape, X_train.dtype)
        print("y_train", y_train.shape, y_train.dtype)
        print("X_test", X_test_sampled.shape, X_test_sampled.dtype)
        print("y_test", y_test_sampled.shape, y_test_sampled.dtype)

        shapley_values = _compute_shapley_values(X_train,
                                                 np.squeeze(y_train),
                                                 X_test_sampled,
                                                 np.squeeze(y_test_sampled), self.k)

        train_data_op = find_dag_node_by_type(OperatorType.TRAIN_DATA, result.dag_node_to_inspection_results)
        inspection_result = result.dag_node_to_inspection_results[train_data_op][lineage_inspection]
        lineage_per_row = list(inspection_result['mlinspect_lineage'])

        fact_table_source = [train_source for train_source in pipeline.train_sources
                             if train_source.source_type == SourceType.FACTS][0]

        shapley_values_by_row_id = {}

        for polynomial, shapley_value in zip(lineage_per_row, shapley_values):
            for entry in polynomial:
                if entry.operator_id == fact_table_source.operator_id:
                    shapley_values_by_row_id[entry.row_id] = shapley_value

        data = fact_table_source.data
        data['__arguseyes__shapley_value'] = \
            data.apply(lambda row: self._add_shapley(row, shapley_values_by_row_id), axis=1)

        return Source(fact_table_source.operator_id, fact_table_source.source_type, data)

#    @staticmethod


    @staticmethod
    def _add_shapley(row, shapley_values_by_row_id):
        polynomial = row['mlinspect_lineage']
        for entry in polynomial:
            if entry.row_id in shapley_values_by_row_id:
                return shapley_values_by_row_id[entry.row_id]
        return 0.0
