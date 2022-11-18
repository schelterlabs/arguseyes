from arguseyes.issues import Issue, IssueDetector
from arguseyes.templates import SourceType, Output
import copy
import logging


class Fairness(IssueDetector):

    # TODO this assumes binary classification
    def detect(self, pipeline, params) -> Issue:

        sensitive_attribute = params['sensitive_attribute']
        privileged_class = params['privileged_class']
        metric = params['metric']
        threshold = params['max_difference']

        fact_table_index, fact_table_source = [
            (index, test_source) for index, test_source in enumerate(pipeline.test_sources)
            if test_source.source_type == SourceType.ENTITIES][0]

        fact_table_lineage = pipeline.test_source_lineage[fact_table_index]

        # Compute group membership per tuple in the test source data
        if sensitive_attribute in fact_table_source.data.columns:
            is_privileged_by_row_id = \
                self._group_membership_from_fact_table(fact_table_source, fact_table_lineage)
        # Compute group membership over a join
        else:
            side_source = None
            side_source_index = None

            for index, test_source in enumerate(pipeline.test_sources):
                if test_source.source_type == SourceType.SIDE_DATA:
                    if sensitive_attribute in test_source.data.columns:
                        side_source = test_source
                        side_source_index = index
                        break

            if side_source is None:
                raise ValueError(f"Cannot find sensitive attribute {sensitive_attribute} in test sources.")

            side_source_lineage = pipeline.test_source_lineage[side_source_index]

            privileged_id = None
            for polynomial, value in zip(side_source_lineage, list(side_source.data[sensitive_attribute])):
                if value == privileged_class:
                    privileged_id = list(polynomial)[0]
                    break

            if privileged_id is None:
                raise ValueError(f"Cannot find privileged class {privileged_class} for "
                                 f"sensitive attribute {sensitive_attribute} in test sources.")

            lineage_x_test = pipeline.output_lineage[Output.X_TEST]
            is_privileged_by_row_id = \
                self._group_membership_from_side_table(fact_table_source.operator_id, lineage_x_test, privileged_id)

        y_pred = pipeline.outputs[Output.Y_PRED]
        lineage_y_pred = pipeline.output_lineage[Output.Y_PRED]

        # Compute the confusion matrix per group
        y_test = pipeline.outputs[Output.Y_TEST]

        privileged_false_negatives = 0
        privileged_true_positives = 0
        privileged_true_negatives = 0
        privileged_false_positives = 0

        disadvantaged_false_negatives = 0
        disadvantaged_true_positives = 0
        disadvantaged_true_negatives = 0
        disadvantaged_false_positives = 0

        for index, polynomial in enumerate(lineage_y_pred):
            for entry in polynomial:
                if entry.operator_id == fact_table_source.operator_id:
                    # Positive ground truth label
                    if y_test[index] == 1.0:
                        if is_privileged_by_row_id[entry.row_id]:
                            if y_pred[index] == 1.0:
                                privileged_true_positives += 1
                            else:
                                privileged_false_negatives += 1
                        else:
                            if y_pred[index] == 1.0:
                                disadvantaged_true_positives += 1
                            else:
                                disadvantaged_false_negatives += 1
                    # Negative ground truth label
                    else:
                        if is_privileged_by_row_id[entry.row_id]:
                            if y_pred[index] == 1.0:
                                privileged_false_positives += 1
                            else:
                                privileged_true_negatives += 1
                        else:
                            if y_pred[index] == 1.0:
                                disadvantaged_false_positives += 1
                            else:
                                disadvantaged_true_negatives += 1


        identifier_privileged = f'arguseyes.fairness.{sensitive_attribute}.{privileged_class.lower()}'
        identifier_disadvantaged = f'arguseyes.fairness.{sensitive_attribute}.not.{privileged_class.lower()}'

        self.log_metric(f'{identifier_privileged}.true_positives', privileged_true_positives)
        self.log_metric(f'{identifier_privileged}.false_negatives', privileged_false_negatives)
        self.log_metric(f'{identifier_privileged}.false_positives', privileged_false_positives)
        self.log_metric(f'{identifier_privileged}.true_negatives', privileged_true_negatives)

        self.log_metric(f'{identifier_disadvantaged}.true_positives', disadvantaged_true_positives)
        self.log_metric(f'{identifier_disadvantaged}.false_negatives', disadvantaged_false_negatives)
        self.log_metric(f'{identifier_disadvantaged}.false_positives', disadvantaged_false_positives)
        self.log_metric(f'{identifier_disadvantaged}.true_negatives', disadvantaged_true_negatives)

        # tp / tp + fp
        predictive_parity = (privileged_true_positives / (privileged_true_positives + privileged_false_positives)) - \
                            (disadvantaged_true_positives / (disadvantaged_true_positives +
                                                             disadvantaged_false_positives))

        # tp / tp + fn
        equal_opportunity = (privileged_true_positives / (privileged_true_positives + privileged_false_negatives)) - \
                            (disadvantaged_true_positives / (disadvantaged_true_positives +
                                                             disadvantaged_false_negatives))

        # tp + fp / tp + fp + tn + fn
        statistical_parity = ((privileged_true_positives + privileged_false_positives) /
                              (privileged_true_positives + privileged_false_negatives + privileged_false_positives +
                               privileged_true_negatives)) - \
                             ((disadvantaged_true_positives + disadvantaged_false_positives) /
                              (disadvantaged_true_positives + disadvantaged_false_negatives +
                               disadvantaged_false_positives + disadvantaged_true_negatives))

        self.log_metric(f'{identifier_privileged}.predictive_parity', predictive_parity)
        self.log_metric(f'{identifier_privileged}.equal_opportunity', equal_opportunity)
        self.log_metric(f'{identifier_privileged}.statistical_parity', statistical_parity)

        metrics = {
            'predictive_parity': predictive_parity,
            'equal_opportunity': equal_opportunity,
            'statistical_parity': statistical_parity
        }

        metric_value = metrics[metric]

        logging.info(f'Fairness metric {metric} for sensitive attribute {sensitive_attribute} ' + \
                     f'with privileged class {privileged_class} is {metric_value}.')

        is_violated = metric_value < -threshold

        issue_details = copy.deepcopy(params)
        issue_details['metric_value'] = metric_value

        return Issue('fairness', is_violated, issue_details)


    def error_msg(self, issue) -> str:
        return f'Found fairness violation for sensitive attribute {issue.details["sensitive_attribute"]} ' + \
               f'with privileged class {issue.details["privileged_class"]}:\nMetric {issue.details["metric"]} ' + \
               f'exceeds max_difference of {issue.details["max_difference"]}' + \
               f' with value {abs(issue.details["metric_value"]):.5f}!'


    def _group_membership_from_fact_table(self, fact_table_source, fact_table_lineage, sensitive_attribute,
                                          privileged_class):
        is_privileged_by_row_id = {}
        for index, row in fact_table_source.data.iterrows():
            is_privileged = row[sensitive_attribute] == privileged_class
            polynomial = fact_table_lineage[index]
            row_id = list(polynomial)[0].row_id
            is_privileged_by_row_id[row_id] = is_privileged
        return is_privileged_by_row_id


    def _group_membership_from_side_table(self, fact_table_operator_id, lineage_x_test, privileged_id):
        is_privileged_by_row_id = {}
        for polynomial_of_row in lineage_x_test:
            is_privileged = privileged_id in polynomial_of_row
            row_id = [entry for entry in polynomial_of_row if entry.operator_id == fact_table_operator_id][0].row_id
            is_privileged_by_row_id[row_id] = is_privileged
        return is_privileged_by_row_id
