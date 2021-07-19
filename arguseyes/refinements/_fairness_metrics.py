import numpy as np

from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.templates.source import SourceType
from arguseyes.utils.dag_extraction import find_dag_node_by_type

from arguseyes.refinements._refinement import Refinement


class FairnessMetrics(Refinement):

    def __init__(self, sensitive_attribute, non_protected_class):
        self.sensitive_attribute = sensitive_attribute
        self.non_protected_class = non_protected_class

    # TODO this assumes binary classification and currently only works attributes of the FACT table
    # TODO this needs some refactoring
    def _compute(self, pipeline):
        result = pipeline.result
        lineage_inspection = pipeline.lineage_inspection

        fact_table_source = [test_source for test_source in pipeline.test_sources
                             if test_source.source_type == SourceType.FACTS][0]

        # Compute group membership per tuple in the test source data
        is_in_non_protected_by_row_id = self._group_membership_in_test_source(fact_table_source)

        # Extract prediction vector for test set
        score_op = find_dag_node_by_type(OperatorType.SCORE, result.dag_node_to_inspection_results)
        predictions_with_lineage = result.dag_node_to_inspection_results[score_op][lineage_inspection]

        y_pred = np.array(predictions_with_lineage['array']).reshape(-1, 1)

        # Compute the confusion matrix per group
        y_test = pipeline.y_test

        non_protected_false_negatives = 0
        non_protected_true_positives = 0
        non_protected_true_negatives = 0
        non_protected_false_positives = 0

        protected_false_negatives = 0
        protected_true_positives = 0
        protected_true_negatives = 0
        protected_false_positives = 0

        for index, polynomial in enumerate(list(predictions_with_lineage['mlinspect_lineage'])):
            for entry in polynomial:
                if entry.operator_id == fact_table_source.operator_id:
                    # Positive ground truth label
                    if y_test[index] == 1.0:
                        if is_in_non_protected_by_row_id[entry.row_id]:
                            if y_pred[index] == 1.0:
                                non_protected_true_positives += 1
                            else:
                                non_protected_false_negatives += 1
                        else:
                            if y_pred[index] == 1.0:
                                protected_true_positives += 1
                            else:
                                protected_false_negatives += 1
                    # Negative ground truth label
                    else:
                        if is_in_non_protected_by_row_id[entry.row_id]:
                            if y_pred[index] == 1.0:
                                non_protected_false_positives += 1
                            else:
                                non_protected_true_negatives += 1
                        else:
                            if y_pred[index] == 1.0:
                                protected_false_positives += 1
                            else:
                                protected_true_negatives += 1

        # Print false negatives rates (as example)
        non_protected_fnr = float(non_protected_false_negatives) / \
                            (float(non_protected_false_negatives) + float(non_protected_true_positives))
        protected_fnr = float(protected_false_negatives) / \
                        (float(protected_false_negatives) + float(protected_true_positives))

        print(f'FNR ({self.sensitive_attribute}={self.non_protected_class}): {non_protected_fnr}, ' +
              f'FNR ({self.sensitive_attribute}!={self.non_protected_class}): {protected_fnr}')

    # TODO return confusion matrices or grouped truth/prediction vectors so that we can rely on sklearn metrics

    def _group_membership_in_test_source(self, fact_table_source):
        is_in_non_protected_by_row_id = {}
        for index, row in fact_table_source.data.iterrows():
            is_in_majority = row[self.sensitive_attribute] == self.non_protected_class
            row_id = list(row['mlinspect_lineage'])[0].row_id
            is_in_non_protected_by_row_id[row_id] = is_in_majority
        return is_in_non_protected_by_row_id