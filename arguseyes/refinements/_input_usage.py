from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.refinements._refinement import Refinement
from arguseyes.utils.dag_extraction import find_dag_node_by_type
from arguseyes.templates.source import Source


class InputUsage(Refinement):

    @staticmethod
    def _is_used(row, lineage_by_source):
        polynomial = row['mlinspect_lineage']
        for entry in polynomial:
            if entry.operator_id in lineage_by_source:
                if entry.row_id in lineage_by_source[entry.operator_id]:
                    return True
        return False

    def _compute(self, pipeline):

        result = pipeline.result
        lineage_inspection = pipeline.lineage_inspection

        train_data_op = find_dag_node_by_type(OperatorType.TRAIN_DATA, result.dag_node_to_inspection_results)
        inspection_result = result.dag_node_to_inspection_results[train_data_op][lineage_inspection]
        lineage_per_row = list(inspection_result['mlinspect_lineage'])

        lineage_by_source = {}

        for polynomial in lineage_per_row:
            for entry in polynomial:
                if entry.operator_id not in lineage_by_source:
                    lineage_by_source[entry.operator_id] = set()

                lineage_by_source[entry.operator_id].add(entry.row_id)

        refined_sources = []

        for source in pipeline.train_sources:
            data = source.data
            data['__arguseyes__is_used'] = data.apply(lambda row: self._is_used(row, lineage_by_source), axis=1)

            refined_source = Source(source.operator_id, source.source_type, data)

            refined_sources.append(refined_source)

        return refined_sources