import numpy as np
from numba import njit, prange

from arguseyes.refinements import Refinement
from arguseyes.templates import SourceType, Source, Output

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


class DataValuation(Refinement):

    def __init__(self, k=1, num_test_samples=10):
        self.k = k
        self.num_test_samples = num_test_samples

    def _compute(self, pipeline):
        X_train = pipeline.outputs[Output.X_TRAIN]
        X_test = pipeline.outputs[Output.X_TEST]
        y_train = pipeline.outputs[Output.Y_TRAIN]
        y_test = pipeline.outputs[Output.Y_TRAIN]

        X_test_sampled = X_test[:self.num_test_samples, :]
        y_test_sampled = y_test[:self.num_test_samples, :]

        shapley_values = _compute_shapley_values(X_train,
                                                 np.squeeze(y_train),
                                                 X_test_sampled,
                                                 np.squeeze(y_test_sampled), self.k)

        lineage_X_train = pipeline.output_lineage[Output.X_TRAIN]    

        fact_table_index, fact_table_source = [(index, test_source) for index, test_source in enumerate(pipeline.test_sources)
                                               if test_source.source_type == SourceType.ENTITIES][0]

        shapley_values_by_row_id = {}

        for polynomial, shapley_value in zip(lineage_X_train, shapley_values):
            for entry in polynomial:
                if entry.operator_id == fact_table_source.operator_id:
                    shapley_values_by_row_id[entry.row_id] = shapley_value

        data = fact_table_source.data
        fact_table_lineage = pipeline.test_source_lineage[fact_table_index]

        for row_index, row in data.iterrows():                
            data.at[row_index, '__arguseyes__shapley_value'] = \
                self._find_shapley(fact_table_lineage[row_index], shapley_values_by_row_id)

        self.log_tag('arguseyes.data_valuation.operator_id', fact_table_source.operator_id)
        self.log_tag('arguseyes.data_valuation.k', self.k)
        self.log_tag('arguseyes.data_valuation.num_test_samples', self.num_test_samples)
        self.log_tag('arguseyes.data_valuation.data_file', 'input-with-shapley-values.parquet')
        self.log_as_parquet_file(data, 'input-with-shapley-values.parquet')

        return Source(fact_table_source.operator_id, fact_table_source.source_type, data)

    @staticmethod
    def _find_shapley(polynomial, shapley_values_by_row_id):
        for entry in polynomial:
            if entry.row_id in shapley_values_by_row_id:
                return shapley_values_by_row_id[entry.row_id]
        return 0.0
