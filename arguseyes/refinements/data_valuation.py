import numpy as np

from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.templates.source import SourceType, Source
from arguseyes.utils.dag_extraction import find_dag_node_by_type


# Shapley value estimation, copied from
# https://gist.github.com/daviddao/e091d66b7a0a3b44486cd0b4035468ee
def get_shapley_value_np(X_train, y_train, X_test, y_test, K=1):
    N = X_train.shape[0]
    M = X_test.shape[0]
    s = np.zeros((N, M))

    for i, (X, y) in enumerate(zip(X_test, y_test)):
        diff = (X_train - X).reshape(N, -1)
        dist = np.einsum('ij, ij->i', diff, diff)
        idx = np.argsort(dist)
        ans = y_train[idx]
        s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
        cur = N - 2
        for j in range(N - 1):
            s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * (min(cur, K - 1) + 1) / (cur + 1)
            cur -= 1
    return np.mean(s, axis=1)


def _add_shapley(row, shapley_values_by_row_id):
    polynomial = row['mlinspect_lineage']
    for entry in polynomial:
        if entry.row_id in shapley_values_by_row_id:
            return shapley_values_by_row_id[entry.row_id]
    return 0.0


def refine(classification_pipeline):
    result = classification_pipeline.result
    lineage_inspection = classification_pipeline.lineage_inspection

    X_train = classification_pipeline.X_train
    X_test = classification_pipeline.X_test
    y_train = classification_pipeline.y_train
    y_test = classification_pipeline.y_test

    k = 1
    num_test_samples = 10

    shapley_values = get_shapley_value_np(X_train, y_train,
                                          X_test[:num_test_samples, :], y_test[:num_test_samples, :],
                                          K=k)

    train_data_op = find_dag_node_by_type(OperatorType.TRAIN_DATA, result.dag_node_to_inspection_results)
    inspection_result = result.dag_node_to_inspection_results[train_data_op][lineage_inspection]
    lineage_per_row = list(inspection_result['mlinspect_lineage'])

    fact_table_source = [train_source for train_source in classification_pipeline.train_sources
                         if train_source.source_type == SourceType.FACTS][0]

    shapley_values_by_row_id = {}

    for polynomial, shapley_value in zip(lineage_per_row, shapley_values):
        for entry in polynomial:
            if entry.operator_id == fact_table_source.operator_id:
                shapley_values_by_row_id[entry.row_id] = shapley_value

    data = fact_table_source.data
    data['__arguseyes__shapley_value'] = data.apply(lambda row: _add_shapley(row, shapley_values_by_row_id), axis=1)

    refined_source = Source(fact_table_source.operator_id, fact_table_source.source_type, data)
    return refined_source
