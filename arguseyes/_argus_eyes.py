import mlflow
from arguseyes.templates.classification import ClassificationPipeline


class ArgusEyes:

    def __init__(self, series_id, artifact_storage_uri):
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(artifact_storage_uri)
        self.mlflow.set_experiment(series_id)

    def classification_pipeline_from_py_file(self, pyfile, cmd_args=[]):
        return ClassificationPipeline.from_py_file(pyfile, cmd_args)
