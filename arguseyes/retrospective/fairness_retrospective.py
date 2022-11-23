import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML


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

    def show_confusion_matrix(self, sensitive_attribute, privileged_class):

        def from_counts(counts, sensitive_attribute, privileged_class, score):
            the_slice = counts
            the_slice = the_slice[the_slice.sensitive_attribute == sensitive_attribute]
            the_slice = the_slice[the_slice.privileged_class == privileged_class]
            the_slice = the_slice[the_slice.sensitive_attribute == sensitive_attribute]
            the_slice = the_slice[the_slice.score == score]
            return int(list(the_slice.count_privileged)[0]), int(list(the_slice.count_disadvantaged)[0])

        counts = self.confusion_matrices_for_groups()

        tn_priv, tn_dis = from_counts(counts, sensitive_attribute, privileged_class, 'true_negatives')
        fp_priv, fp_dis = from_counts(counts, sensitive_attribute, privileged_class, 'false_positives')
        fn_priv, fn_dis = from_counts(counts, sensitive_attribute, privileged_class, 'false_negatives')
        tp_priv, tp_dis = from_counts(counts, sensitive_attribute, privileged_class, 'true_positives')

        html = f"""
        <div style="display:flex; margin-left:-5px; margin-right:-5px; width: 85%;">
          <div style="flex: 50%; padding: 5px;">
            <h5>Confusion matrix for the privileged group ({sensitive_attribute}={privileged_class})</h5>
            <table>
              <tr>
                <td></td><td><b>positive prediction</b></td><td><b>negative prediction</b></td><td></td>
              </tr>
              <tr>
                <td><b>positive label</b></td>
                <td style="border-left: 2px solid black;border-top: 2px solid black;">TP={tp_priv}</td>
                <td style="border-right: 2px solid black;border-top: 2px solid black;">FN={fn_priv}</td>
                <td>{tp_priv + fn_priv}</td>
              </tr>    
              <tr>
                <td><b>negative label</b></td>
                <td style="border-left: 2px solid black;border-bottom: 2px solid black;">FP={fp_priv}</td>
                <td style="border-right: 2px solid black;border-bottom: 2px solid black;">TN={tn_priv}</td>
                <td>{fp_priv + tn_priv}</td>
              </tr>  
              <tr>
                <td></td><td>{tp_priv + fp_priv}</td><td>{fn_priv + tn_priv}</td><td></td>
              </tr>    
            </table>
          </div>
          <div style="flex: 50%; padding: 5px;">
            <h5>Confusion matrix for the disadvantaged group ({sensitive_attribute}!={privileged_class})</h5>  
            <table>
              <tr>
                <td></td><td><b>positive prediction</b></td><td><b>negative prediction</b></td><td></td>
              </tr>
              <tr>
                <td><b>positive label</b></td>
                <td style="border-left: 2px solid black;border-top: 2px solid black;">TP={tp_dis}</td>
                <td style="border-right: 2px solid black;border-top: 2px solid black;">FN={fn_dis}</td>
                <td>{tp_dis + fn_dis}</td>
              </tr>    
              <tr>
                <td><b>negative label</b></td>
                <td style="border-left: 2px solid black;border-bottom: 2px solid black;">FP={fp_dis}</td>
                <td style="border-right: 2px solid black;border-bottom: 2px solid black;">TN={tn_dis}</td>
                <td>{fp_dis + tn_dis}</td>
              </tr>  
              <tr>
                <td></td><td>{tp_dis + fp_dis}</td><td>{fn_dis + tn_dis}</td><td></td>
              </tr>    
            </table>
          </div>  
        </div>  
        """

        display(HTML(html))

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
