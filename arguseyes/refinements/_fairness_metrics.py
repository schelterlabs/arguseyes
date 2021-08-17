import numpy as np

from arguseyes.refinements import Refinement
from arguseyes.templates import SourceType, Output


class FairnessMetrics(Refinement):

    def __init__(self, sensitive_attribute, non_protected_class):
        self.sensitive_attribute = sensitive_attribute
        self.non_protected_class = non_protected_class

    # TODO this assumes binary classification and currently only works attributes of the FACT table
    # TODO this needs some refactoring
    def _compute(self, pipeline):
        fact_table_index, fact_table_source = [(index, test_source) for index, test_source in enumerate(pipeline.test_sources)
                                               if test_source.source_type == SourceType.ENTITIES][0]

        fact_table_lineage = pipeline.test_source_lineage[fact_table_index]    

        # Compute group membership per tuple in the test source data
        is_in_non_protected_by_row_id = self._group_membership_in_test_source(fact_table_source, fact_table_lineage)

        y_pred = pipeline.outputs[Output.Y_PRED]
        lineage_y_pred = pipeline.output_lineage[Output.Y_PRED]

        # Compute the confusion matrix per group
        y_test = pipeline.outputs[Output.Y_TEST]

        non_protected_false_negatives = 0
        non_protected_true_positives = 0
        non_protected_true_negatives = 0
        non_protected_false_positives = 0

        protected_false_negatives = 0
        protected_true_positives = 0
        protected_true_negatives = 0
        protected_false_positives = 0

        for index, polynomial in enumerate(lineage_y_pred):
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

        identifier_non_protected = f'arguseyes.fairness.{self.sensitive_attribute}.{self.non_protected_class.lower()}'
        identifier_protected = f'arguseyes.fairness.{self.sensitive_attribute}.not.{self.non_protected_class.lower()}'

        self.log_metric(f'{identifier_non_protected}.true_positives', non_protected_true_positives)
        self.log_metric(f'{identifier_non_protected}.false_negatives', non_protected_false_negatives)
        self.log_metric(f'{identifier_non_protected}.false_positives', non_protected_false_positives)
        self.log_metric(f'{identifier_non_protected}.true_negatives', non_protected_true_negatives)

        self.log_metric(f'{identifier_protected}.true_positives', protected_true_positives)
        self.log_metric(f'{identifier_protected}.false_negatives', protected_false_negatives)
        self.log_metric(f'{identifier_protected}.false_positives', protected_false_positives)
        self.log_metric(f'{identifier_protected}.true_negatives', protected_true_negatives)

        # False negative rates (as example)
        non_protected_fnr = float(non_protected_false_negatives) / \
                            (float(non_protected_false_negatives) + float(non_protected_true_positives))
        protected_fnr = float(protected_false_negatives) / \
                        (float(protected_false_negatives) + float(protected_true_positives))

        self.log_metric(f'{identifier_non_protected}.false_negative_rate', non_protected_fnr)
        self.log_metric(f'{identifier_protected}.false_negative_rate', protected_fnr)

        # TODO compute more metrics

    def _group_membership_in_test_source(self, fact_table_source, fact_table_lineage):
        is_in_non_protected_by_row_id = {}
        for index, row in fact_table_source.data.iterrows():
            is_in_majority = row[self.sensitive_attribute] == self.non_protected_class
            polynomial = fact_table_lineage[index]
            row_id = list(polynomial)[0].row_id
            is_in_non_protected_by_row_id[row_id] = is_in_majority
        return is_in_non_protected_by_row_id
