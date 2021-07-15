from mlinspect.inspections._inspection_input import OperatorType


# Find the first node of a given type in the DAG
def find_dag_node_by_type(op_type, dag_node_to_inspection_results):
    for node in dag_node_to_inspection_results.keys():
        if node.operator_info.operator == op_type:
            return node

    raise ValueError('Unable to find DAG node')


def find_source_datasets(start_node_id, result, lineage_inspection):
    nodes_to_search = []
    nodes_processed = set()

    source_datasets = {}

    nodes_to_search.append(start_node_id)

    while len(nodes_to_search) > 0:
        current_node_id = nodes_to_search.pop()
        for source, target in result.dag.edges:
            if target.node_id == current_node_id:
                if source.node_id not in nodes_processed and source.node_id not in nodes_to_search:
                    nodes_to_search.append(source.node_id)
                    if source.operator_info.operator == OperatorType.DATA_SOURCE:
                        data = result.dag_node_to_inspection_results[source][lineage_inspection]
                        source_datasets[source.node_id] = data
        nodes_processed.add(current_node_id)

    return source_datasets
