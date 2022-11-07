import pandas as pd
import mlflow


class LabelErrorsRetrospective:

    def __init__(self, pipeline_run):
        self.pipeline_run = pipeline_run
        self.mlflow_run = mlflow.get_run(run_id=pipeline_run.run.info.run_id)

    def load_entity_table_with_shapley_values(self):
        path = f'{self.mlflow_run.info.artifact_uri}/{self.mlflow_run.data.tags["arguseyes.shapley_values.data_file"]}'
        return pd.read_parquet(path)