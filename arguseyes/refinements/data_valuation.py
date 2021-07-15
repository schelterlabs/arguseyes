import numpy as np

from mlinspect.inspections._inspection_input import OperatorType
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


#TODO duplicated, refactor
def _find_source_data_with_lineage(start_node_id, result, lineage_inspection):
    nodes_to_search = []
    nodes_processed = set()

    source_datasets = []

    nodes_to_search.append(start_node_id)

    while len(nodes_to_search) > 0:
        current_node_id = nodes_to_search.pop()
        for source, target in result.dag.edges:
            if target.node_id == current_node_id:
                if source.node_id not in nodes_processed and source.node_id not in nodes_to_search:
                    nodes_to_search.append(source.node_id)
                    if source.operator_info.operator == OperatorType.DATA_SOURCE:
                        data_with_lineage = result.dag_node_to_inspection_results[source][lineage_inspection]

                        source_datasets.append(data_with_lineage)

        nodes_processed.add(current_node_id)

    return source_datasets


def _add_shapley(row, shapley_values_by_source_and_row_id):
    polynomial = row['mlinspect_lineage']
    for entry in polynomial:
        if entry.operator_id in shapley_values_by_source_and_row_id:
            if entry.row_id in shapley_values_by_source_and_row_id[entry.operator_id]:
                return shapley_values_by_source_and_row_id[entry.operator_id][entry.row_id]
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

    shapley_values_by_source_and_row_id = {}

    #TODO this is probably wrong, we need to know how to distribute the shapley values correctly over joins
    for polynomial, shapley_value in zip(lineage_per_row, shapley_values):
        # TODO how to distribute them for more complex lineage?
        for entry in polynomial:
            if entry.operator_id not in shapley_values_by_source_and_row_id:
                shapley_values_by_source_and_row_id[entry.operator_id] = {}

            shapley_values_by_source_and_row_id[entry.operator_id][entry.row_id] = shapley_value

    # TODO we need to detect the "star schema" here

    source_datasets_with_lineage = \
        _find_source_data_with_lineage(train_data_op.node_id, classification_pipeline.result,
                                       classification_pipeline.lineage_inspection)

    refined_datasets = []

    for data_with_lineage in source_datasets_with_lineage:
        data = data_with_lineage.copy(deep=True)

        data['__argos__shapley_value'] = \
            data.apply(lambda row: _add_shapley(row, shapley_values_by_source_and_row_id), axis=1)
        data.drop(columns=['mlinspect_lineage'], inplace=True)

        refined_datasets.append(data)

    return refined_datasets
