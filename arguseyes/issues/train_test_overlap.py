from mlinspect.inspections._inspection_input import OperatorType
from arguseyes.utils.dag_extraction import find_dag_node_by_type


def detect(classification_pipeline) -> bool:

    result = classification_pipeline.result
    lineage_inspection = classification_pipeline.lineage_inspection

    # TODO the lineage should be readily available
    train_data_op = find_dag_node_by_type(OperatorType.TRAIN_DATA, result.dag_node_to_inspection_results)
    train_data_with_lineage = result.dag_node_to_inspection_results[train_data_op][lineage_inspection]

    test_data_op = find_dag_node_by_type(OperatorType.TEST_DATA, result.dag_node_to_inspection_results)
    test_data_with_lineage = result.dag_node_to_inspection_results[test_data_op][lineage_inspection]

    train_lineage = frozenset([frozenset(lineage) for lineage in train_data_with_lineage['mlinspect_lineage']])
    test_lineage = frozenset([frozenset(lineage) for lineage in test_data_with_lineage['mlinspect_lineage']])

    # TODO output tuples in the future
    return len(train_lineage & test_lineage) > 0
