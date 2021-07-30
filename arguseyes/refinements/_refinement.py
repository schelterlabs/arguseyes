from abc import ABC, abstractmethod
import os
import mlflow
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

class Refinement(ABC):
    @abstractmethod
    def _compute(self, pipeline):
        raise NotImplementedError

    def log_metric(self, metric_name, metric_value):
        mlflow.log_metric(metric_name, metric_value)

    def log_tag(self, tag_name, tag_value):
        mlflow.set_tag(tag_name, tag_value)

    def log_as_parquet_file(self, dataframe, filename_to_assign):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, filename_to_assign)
            table = pa.Table.from_pandas(dataframe, preserve_index=True)
            pq.write_table(table, temp_filename)
            mlflow.log_artifact(temp_filename)
