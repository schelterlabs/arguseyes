from mlinspect.inspections._inspection_input import OperatorType
from arguseyes.utils.dag_extraction import find_dag_node_by_type
from arguseyes.utils.tensors import copy_to_matrix


def extract_train_feature_matrix(result, lineage_inspection):
    return _extract(OperatorType.TRAIN_DATA, result, lineage_inspection)


def extract_train_labels(result, lineage_inspection):
    return _extract(OperatorType.TRAIN_LABELS, result, lineage_inspection)


def extract_test_feature_matrix(result, lineage_inspection):
    return _extract(OperatorType.TEST_DATA, result, lineage_inspection)


def extract_test_labels(result, lineage_inspection):
    return _extract(OperatorType.TEST_LABELS, result, lineage_inspection)


def _extract(operator_type, result, lineage_inspection):
    data_op = find_dag_node_by_type(operator_type, result.dag_node_to_inspection_results)
    return copy_to_matrix(result.dag_node_to_inspection_results[data_op][lineage_inspection]['array'].values)

