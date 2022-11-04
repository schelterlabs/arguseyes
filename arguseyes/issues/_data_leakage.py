from arguseyes.issues import Issue, IssueDetector
from arguseyes.templates import Output


class DataLeakage(IssueDetector):

    def detect(self, pipeline, params) -> Issue:

        train_lineage = frozenset([frozenset(lineage) for lineage in pipeline.output_lineage[Output.X_TRAIN]])
        test_lineage = frozenset([frozenset(lineage) for lineage in pipeline.output_lineage[Output.X_TEST]])

        overlap_lineage = train_lineage & test_lineage
        num_leaked_records = len(overlap_lineage)
        has_leakage = num_leaked_records > 0

        # TODO maybe output tuple ids in the future
        return Issue('data_leakage', has_leakage, {'num_leaked_records': num_leaked_records})

    def error_msg(self, issue) -> str:
        return f'Found {issue.id}, {issue.details["num_leaked_records"]} records present in both train and test set!'
