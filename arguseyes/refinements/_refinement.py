from abc import ABC, abstractmethod
import mlflow

class Refinement(ABC):
    @abstractmethod
    def _compute(self, pipeline):
        raise NotImplementedError

    def log_metric(self, metric_name, metric_value):
        mlflow.log_metric(metric_name, metric_value)
