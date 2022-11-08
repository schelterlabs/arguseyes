from arguseyes.issues import Issue, IssueDetector
from arguseyes.templates import Output


class DataLeakage(IssueDetector):

    def detect(self, pipeline, params) -> Issue:

        train_provenance= frozenset([frozenset(provenance) for provenance in pipeline.output_lineage[Output.X_TRAIN]])
        test_provenance = frozenset([frozenset(provenance) for provenance in pipeline.output_lineage[Output.X_TEST]])

        leaked = train_provenance & test_provenance
        num_leaked_records = len(leaked)
        has_leakage = num_leaked_records > 0

        if has_leakage:
            self.log_tag('arguseyes.data_leakage.provenance_file', 'leaked_tuples.pickle')
            self.log_as_pickle_file(leaked, 'leaked_tuples.pickle')

        return Issue('data_leakage', has_leakage, {'num_leaked_records': num_leaked_records})

    def error_msg(self, issue) -> str:
        return f'Found {issue.id}, {issue.details["num_leaked_records"]} records present in both train and test set!'
