import logging
from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.templates import Source, SourceType
from arguseyes.extraction.dag_extraction import find_dag_node_by_type, find_source_datasets
from arguseyes.extraction.heuristics.fact_table_from_star_schema import determine_fact_table_source_id


def extract_train_sources(dag, dag_node_to_lineage_df):
    return _extract_sources(OperatorType.TRAIN_DATA, dag, dag_node_to_lineage_df)


def extract_test_sources(dag, dag_node_to_lineage_df):
    return _extract_sources(OperatorType.TEST_DATA, dag, dag_node_to_lineage_df)


def _extract_sources(operator_type, dag, dag_node_to_lineage_df):
    data_op = find_dag_node_by_type(operator_type, dag_node_to_lineage_df.keys())
    raw_sources = find_source_datasets(data_op.node_id, dag, dag_node_to_lineage_df)

    fact_table_source_id = determine_fact_table_source_id(raw_sources, data_op, dag_node_to_lineage_df)

    sources = []
    source_lineage = []

    for source_id, data in raw_sources.items():
 
        lineage = list(data['mlinspect_lineage'])
        data = data.drop(columns=['mlinspect_lineage'], inplace=False)

        source_lineage.append(lineage)

        if source_id == fact_table_source_id:
            logging.info(f'Found fact table from operator {source_id} with {len(data)} records and ' +
                         f'the following attributes: {data.columns.values.tolist()}')
            sources.append(Source(source_id, SourceType.ENTITIES, data))
        else:
            logging.info(f'Found dimension table from operator {source_id} with {len(data)} records and ' +
                         f'the following attributes: {data.columns.values.tolist()}')
            sources.append(Source(source_id, SourceType.SIDE_DATA, data))

    return sources, source_lineage
