from mlinspect.inspections._inspection_input import OperatorType


def _sources_with_one_to_one_correspondence_to_feature_vectors(feature_matrix_lineage_per_row):
    rows_from_operator = {}
    operators_with_duplicates = set()

    for polynomial in feature_matrix_lineage_per_row:
        for entry in polynomial:

            if entry.operator_id not in operators_with_duplicates:
                if entry.operator_id not in rows_from_operator:
                    rows_from_operator[entry.operator_id] = set()

                if entry.row_id in rows_from_operator[entry.operator_id]:
                    operators_with_duplicates.add(entry.operator_id)
                else:
                    rows_from_operator[entry.operator_id].add(entry.row_id)

    return set(rows_from_operator.keys()).difference(operators_with_duplicates)


# TODO Rewrite this not use the join outputs but only the DAG, to avoid materialisations
def _sources_with_max_join_usage(dag_node_to_lineage_df):
    joins = [node for node in dag_node_to_lineage_df.keys()
             if node.operator_info.operator == OperatorType.JOIN]

    source_join_usage_counts = {}

    for join in joins:
        join_output_with_lineage = dag_node_to_lineage_df[join]
        lineage_per_row = list(join_output_with_lineage['mlinspect_lineage'])

        source_ids = set()

        for polynomial in lineage_per_row:
            for entry in polynomial:
                source_ids.add(entry.operator_id)

        for source_id in source_ids:
            if source_id not in source_join_usage_counts:
                source_join_usage_counts[source_id] = 0

            source_join_usage_counts[source_id] += 1

    max_usage = max(source_join_usage_counts.values())
    sources_with_max_usage = set([source_id for source_id, count in source_join_usage_counts.items() \
                                  if count == max_usage])

    return sources_with_max_usage


def _sources_with_max_cardinality(raw_sources):
    max_cardinality = max([len(source_data) for _, source_data in raw_sources.items()])
    max_cardinality_sources = set(
        [source_id for source_id, source_data in raw_sources.items() if len(source_data) == max_cardinality])
    return max_cardinality_sources


# TODO handle errors
def determine_fact_table_source_id(raw_sources, data_op, dag_node_to_lineage_df):
    # Heuristic 1: Fact table should have 1:1 correspondence between input tuples and features
    # TODO this is not the case for fork-pipelines!
    feature_matrix_lineage = dag_node_to_lineage_df[data_op]
    feature_matrix_lineage_per_row = list(feature_matrix_lineage['mlinspect_lineage'])
    sources_one_to_one = _sources_with_one_to_one_correspondence_to_feature_vectors(feature_matrix_lineage_per_row)

    if len(sources_one_to_one) == 1:
        return list(sources_one_to_one)[0]
    else:
        # Heuristic 2: Fact table is most often used in joins
        sources_max_usage = _sources_with_max_join_usage(dag_node_to_lineage_df)
        remaining_sources = sources_one_to_one & sources_max_usage

        if len(remaining_sources) == 1:
            return list(remaining_sources)[0]
        else:
            # Heuristic 3: Fact table is largest input
            sources_max_cardinality = _sources_with_max_cardinality(raw_sources)
            remaining_sources = remaining_sources & sources_max_cardinality
        if len(remaining_sources) == 1:
            return list(remaining_sources)[0]
        else:
            # Heuristic 4: Fact table is loaded first...
            return min(remaining_sources)
