from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.templates.source import Source, SourceType
from arguseyes.utils.dag_extraction import find_dag_node_by_type, find_source_datasets
from arguseyes.templates.heuristics.fact_table_from_star_schema import determine_fact_table_source_id


def extract_train_sources(result, lineage_inspection):
    return _extract_sources(OperatorType.TRAIN_DATA, result, lineage_inspection)


def extract_test_sources(result, lineage_inspection):
    return _extract_sources(OperatorType.TEST_DATA, result, lineage_inspection)


def _extract_sources(operator_type, result, lineage_inspection):
    data_op = find_dag_node_by_type(operator_type, result.dag_node_to_inspection_results)
    raw_sources = find_source_datasets(data_op.node_id, result, lineage_inspection)

    fact_table_source_id = determine_fact_table_source_id(raw_sources, data_op, result, lineage_inspection)

    sources = []

    for source_id, data in raw_sources.items():

        if source_id == fact_table_source_id:
            sources.append(Source(source_id, SourceType.FACTS, data))
        else:
            sources.append(Source(source_id, SourceType.DIMENSION, data))

    return sources
