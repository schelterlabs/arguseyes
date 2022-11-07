import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class FairnessRetrospective:

    def __init__(self, pipeline_run):
        self.pipeline_run = pipeline_run
        self.mlflow_run = mlflow.get_run(run_id=pipeline_run.run.info.run_id)

    def fairness_criteria(self):
        criteria = set()
        for key in self.mlflow_run.data.metrics:
            if key.startswith('arguseyes.fairness') and not '.not.' in key:
                criterion = (key.split('.')[2], key.split('.')[3])
                criteria.add(criterion)
        return criteria

    def confusion_matrices_for_groups(self):
        criteria = self.fairness_criteria()

        records = []

        for sensitive_attribute, privileged_class in criteria:
            for score in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']:
                count_priv = \
                    self.mlflow_run.data.metrics[f'arguseyes.fairness.{sensitive_attribute}.{privileged_class}.{score}']
                count_dis = \
                    self.mlflow_run.data.metrics[
                        f'arguseyes.fairness.{sensitive_attribute}.not.{privileged_class}.{score}']

                records.append((sensitive_attribute, privileged_class, score, count_priv, count_dis))

        columns = ['sensitive_attribute', 'privileged_class', 'score', 'count_privileged', 'count_disadvantaged']
        confusion_matrices = pd.DataFrame.from_records(records, columns=columns)

        return confusion_matrices

    def fairness_metrics(self, sensitive_attribute, privileged_class):
        tp_priv = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.{privileged_class}.true_positives']
        fn_priv = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.{privileged_class}.false_negatives']
        tn_priv = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.{privileged_class}.true_negatives']
        fp_priv = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.{privileged_class}.false_positives']

        tp_dis = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.not.{privileged_class}.true_positives']
        fn_dis = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.not.{privileged_class}.false_negatives']
        tn_dis = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.not.{privileged_class}.true_negatives']
        fp_dis = self.mlflow_run.data.metrics[
            f'arguseyes.fairness.{sensitive_attribute}.not.{privileged_class}.false_positives']

        # tp / tp + fp
        predictive_parity = (tp_priv / (tp_priv + fp_priv)) - (tp_dis / (tp_dis + fp_dis))

        # tp / tp + fn
        equal_opportunity = (tp_priv / (tp_priv + fn_priv)) - (tp_dis / (tp_dis + fn_dis))

        # tp + fp / tp + fp + tn + fn
        statistical_parity = ((tp_priv + fp_priv) / (tp_priv + fn_priv + fp_priv + tn_priv)) \
                             - ((tp_dis + fp_dis) / (tp_dis + fn_dis + fp_dis + tn_dis))

        return {
            'predictive_parity': predictive_parity,
            'equal_opportunity': equal_opportunity,
            'statistical_parity': statistical_parity
        }

    def plot_fairness_metrics(self, sensitive_attribute, non_protected_class):

        metrics = self.fairness_metrics(sensitive_attribute, non_protected_class)

        max_value = np.max(np.absolute(list(metrics.values())))
        plt.ylim((-max_value * 1.25, max_value * 1.25))

        plt.bar(np.arange(3),
                [metrics['predictive_parity'], metrics['equal_opportunity'], metrics['statistical_parity']])

        plt.xticks(np.arange(3), ['predictive parity', 'equal opportunity', 'statistical parity'])
        plt.show()
