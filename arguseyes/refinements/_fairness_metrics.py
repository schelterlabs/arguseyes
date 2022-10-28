from arguseyes.refinements import Refinement
from arguseyes.templates import SourceType, Output


class FairnessMetrics(Refinement):

    def __init__(self, sensitive_attribute, non_protected_class):
        self.sensitive_attribute = sensitive_attribute
        self.non_protected_class = non_protected_class

    # TODO this assumes binary classification
    def _compute(self, pipeline):
        fact_table_index, fact_table_source = [
            (index, test_source) for index, test_source in enumerate(pipeline.test_sources)
            if test_source.source_type == SourceType.ENTITIES][0]

        fact_table_lineage = pipeline.test_source_lineage[fact_table_index]

        # Compute group membership per tuple in the test source data
        if self.sensitive_attribute in fact_table_source.data.columns:
            is_in_non_protected_by_row_id = \
                self._group_membership_from_fact_table(fact_table_source, fact_table_lineage)
        # Compute group membership over a join
        else:
            side_source = None
            side_source_index = None

            for index, test_source in enumerate(pipeline.test_sources):
                if test_source.source_type == SourceType.SIDE_DATA:
                    if self.sensitive_attribute in test_source.data.columns:
                        side_source = test_source
                        side_source_index = index
                        break

            if side_source is None:
                raise ValueError(f"Cannot find sensitive attribute {self.sensitive_attribute} in test sources.")

            side_source_lineage = pipeline.test_source_lineage[side_source_index]

            non_protected_id = None
            for polynomial, value in zip(side_source_lineage, list(side_source.data[self.sensitive_attribute])):
                if value == self.non_protected_class:
                    non_protected_id = list(polynomial)[0]
                    break

            if non_protected_id is None:
                raise ValueError(f"Cannot find non-protected class {self.non_protected_class} for "
                                 f"sensitive attribute {self.sensitive_attribute} in test sources.")

            lineage_x_test = pipeline.output_lineage[Output.X_TEST]
            is_in_non_protected_by_row_id = \
                self._group_membership_from_side_table(fact_table_source.operator_id, lineage_x_test, non_protected_id)

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

    def _group_membership_from_fact_table(self, fact_table_source, fact_table_lineage):
        is_in_non_protected_by_row_id = {}
        for index, row in fact_table_source.data.iterrows():
            is_in_non_protected = row[self.sensitive_attribute] == self.non_protected_class
            polynomial = fact_table_lineage[index]
            row_id = list(polynomial)[0].row_id
            is_in_non_protected_by_row_id[row_id] = is_in_non_protected
        return is_in_non_protected_by_row_id


    def _group_membership_from_side_table(self, fact_table_operator_id, lineage_x_test, non_protected_id):
        is_in_non_protected_by_row_id = {}
        for polynomial_of_row in lineage_x_test:
            is_in_non_protected = non_protected_id in polynomial_of_row
            row_id = [entry for entry in polynomial_of_row if entry.operator_id == fact_table_operator_id][0].row_id
            is_in_non_protected_by_row_id[row_id] = is_in_non_protected
        return is_in_non_protected_by_row_id
