# https://github.com/mlflow/mlp-regression-example/
import logging

from typing import Dict, Any
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

_logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:
    if file_format == "csv":
        import pandas
        return pandas.read_csv(file_path)
    elif file_format == "parquet":
        import pandas
        return pandas.read_parquet(file_path)
    else:
        raise NotImplementedError


def filter_dataset(dataset: DataFrame):
    filtered_dataset = dataset.dropna()
    filtered_dataset = filtered_dataset[filtered_dataset["fare_amount"] > 0]
    filtered_dataset = filtered_dataset[filtered_dataset["trip_distance"] < 400]
    filtered_dataset = filtered_dataset[filtered_dataset["trip_distance"] > 0]
    filtered_dataset = filtered_dataset[filtered_dataset["fare_amount"] < 1000]
    return filtered_dataset


def calculate_features(df: DataFrame):
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    trip_duration = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df["trip_duration"] = trip_duration.map(lambda x: x.total_seconds() / 60)
    dateTimeColumns = list(df.select_dtypes(include=['datetime64']).columns)
    for dateTimeColumn in dateTimeColumns:
        df[dateTimeColumn] = df[dateTimeColumn].astype(str)
    df.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"], inplace=True)
    return df


def transformer_fn():
    return Pipeline(
        steps=[
            ("calculate_time_and_duration_features", FunctionTransformer(calculate_features)),
            ("encoder", ColumnTransformer(
                transformers=[
                    ("hour_encoder", OneHotEncoder(categories="auto", sparse=False), ["pickup_hour"],),
                    ("day_encoder", OneHotEncoder(categories="auto", sparse=False), ["pickup_dow"],),
                    ("std_scaler", StandardScaler(), ["trip_distance", "trip_duration"],),]),),
        ]
    )


def estimator_fn(estimator_params: Dict[str, Any] = {}):
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor(random_state=42, **estimator_params)


data = load_file_as_dataframe('datasets/nyc-taxi/sample.parquet', 'parquet')
filtered_data = filter_dataset(data)

temporal_split_date = pd.to_datetime('2016-02-15')

train_data = filtered_data[filtered_data['tpep_dropoff_datetime'].dt.date <= temporal_split_date]
test_data = filtered_data[filtered_data['tpep_dropoff_datetime'].dt.date >= temporal_split_date]
# FIX for leakage: make sure that train/test are disjunct by changing the predicate
# test_data = filtered_data[filtered_data['tpep_dropoff_datetime'].dt.date > temporal_split_date]

model = Pipeline([
    ('featurization', transformer_fn()),
    ('learner', estimator_fn())
])

model.fit(train_data, train_data['fare_amount'])
print(model.score(test_data, test_data['fare_amount']))