from mlinspect.inspections._inspection_input import OperatorType
from arguseyes.extraction.dag_extraction import find_dag_node_by_type
from arguseyes.extraction.tensors import copy_to_matrix


def extract_train_feature_matrix(dag_node_to_lineage_df):
    return _extract(OperatorType.TRAIN_DATA, dag_node_to_lineage_df)


def extract_train_labels(dag_node_to_lineage_df):
    return _extract(OperatorType.TRAIN_LABELS, dag_node_to_lineage_df)


def extract_test_feature_matrix(dag_node_to_lineage_df):
    return _extract(OperatorType.TEST_DATA, dag_node_to_lineage_df)


def extract_test_labels(dag_node_to_lineage_df):
    return _extract(OperatorType.TEST_LABELS, dag_node_to_lineage_df)

def extract_predicted_labels(dag_node_to_lineage_df):
    return _extract(OperatorType.SCORE, dag_node_to_lineage_df)

def _extract(operator_type, dag_node_to_lineage_df):
    data_op = find_dag_node_by_type(operator_type, dag_node_to_lineage_df.keys())
    matrix = copy_to_matrix(dag_node_to_lineage_df[data_op]['array'].values)
    lineage = list(dag_node_to_lineage_df[data_op]['mlinspect_lineage'])
    return matrix, lineage
