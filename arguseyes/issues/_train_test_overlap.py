from arguseyes.issues import Issue, IssueDetector
from arguseyes.templates import Output

class TrainTestOverlap(IssueDetector):

    def _detect(self, pipeline) -> Issue:

        train_lineage = frozenset([frozenset(lineage) for lineage in pipeline.output_lineage[Output.X_TRAIN]])
        test_lineage = frozenset([frozenset(lineage) for lineage in pipeline.output_lineage[Output.X_TEST]])

        overlap_lineage = train_lineage & test_lineage
        num_overlapping_records = len(overlap_lineage)
        has_overlap = num_overlapping_records > 0

        # TODO maybe output tuple ids in the future
        return Issue('traintest_overlap', has_overlap, {'num_overlapping_records': num_overlapping_records})

    def error_msg(self, issue) -> str:
        return f'Found {issue.id}, {issue.details["num_overlapping_records"]} records present in both train and test set!'
