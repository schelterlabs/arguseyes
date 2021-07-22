from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.issues import Issue
from arguseyes.issues._issue import IssueDetector
from arguseyes.utils.dag_extraction import find_dag_node_by_type


class TrainTestOverlap(IssueDetector):

    def _detect(self, pipeline) -> Issue:

        # TODO the lineage should be readily available
        train_data_op = find_dag_node_by_type(OperatorType.TRAIN_DATA, pipeline.dag_node_to_lineage_df.keys())
        train_data_with_lineage = pipeline.dag_node_to_lineage_df[train_data_op]

        test_data_op = find_dag_node_by_type(OperatorType.TEST_DATA, pipeline.dag_node_to_lineage_df.keys())
        test_data_with_lineage = pipeline.dag_node_to_lineage_df[test_data_op]

        train_lineage = frozenset([frozenset(lineage) for lineage in train_data_with_lineage['mlinspect_lineage']])
        test_lineage = frozenset([frozenset(lineage) for lineage in test_data_with_lineage['mlinspect_lineage']])

        overlap_lineage = train_lineage & test_lineage
        num_overlapping_records = len(overlap_lineage)
        has_overlap = num_overlapping_records > 0

        # TODO maybe output tuple ids in the future
        return Issue('traintest_overlap', has_overlap, {'num_overlapping_records': num_overlapping_records})
