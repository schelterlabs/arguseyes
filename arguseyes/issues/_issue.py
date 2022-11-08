from __future__ import annotations
import dataclasses
from abc import ABC, abstractmethod

import os
import tempfile

import mlflow
import pyarrow as pa
import pyarrow.parquet as pq
import pickle

@dataclasses.dataclass
class Issue:
    id: str
    is_present: bool
    details: dict


class IssueDetector(ABC):
    @abstractmethod
    def detect(self, pipeline, params) -> Issue:
        raise NotImplementedError

    def error_msg(self, issue) -> str:
        return f'Found {issue.id}'

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

    def log_as_pickle_file(self, object, filename_to_assign):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, filename_to_assign)
            with open(temp_filename, 'wb') as out_file:
                pickle.dump(object, out_file, protocol=pickle.HIGHEST_PROTOCOL)
            mlflow.log_artifact(temp_filename)
