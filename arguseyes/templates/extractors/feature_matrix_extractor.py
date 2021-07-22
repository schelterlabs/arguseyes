from mlinspect.inspections._inspection_input import OperatorType
from arguseyes.utils.dag_extraction import find_dag_node_by_type
from arguseyes.utils.tensors import copy_to_matrix


def extract_train_feature_matrix(result):
    return _extract(OperatorType.TRAIN_DATA, result)


def extract_train_labels(result):
    return _extract(OperatorType.TRAIN_LABELS, result)


def extract_test_feature_matrix(result):
    return _extract(OperatorType.TEST_DATA, result)


def extract_test_labels(result):
    return _extract(OperatorType.TEST_LABELS, result)


def _extract(operator_type, result):
    data_op = find_dag_node_by_type(operator_type, result.dag_node_to_inspection_results)
    return copy_to_matrix(tuple(result.dag_node_to_inspection_results[data_op])[1]['array'].values)

