import numpy as np

from mlinspect.inspections._inspection_input import OperatorType
from arguseyes.extraction.dag_extraction import find_dag_node_by_type


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
    captured_intermediate = dag_node_to_lineage_df[data_op]

    columns = [column for column in captured_intermediate.columns if column != 'mlinspect_lineage']
    # There should only be one column, either 'array' for numpy arrays, or a named column for dataframe inputs
    column_of_interest = columns[0]

    matrix = np.vstack(captured_intermediate[column_of_interest].values)
    lineage = list(captured_intermediate['mlinspect_lineage'])
    return matrix, lineage
