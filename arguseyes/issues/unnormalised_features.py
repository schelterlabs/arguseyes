import numpy as np


def _is_binary(column):
    unique_values = np.unique(column)
    return len(unique_values) == 2 and 0.0 in unique_values and 1.0 in unique_values


def _has_zero_mean(column):
    return np.isclose(np.mean(column), 0.0)


def _has_unit_variance(column):
    return np.isclose(np.var(column), 1.0)


def detect(classification_pipeline) -> bool:

    X_train = classification_pipeline.X_train
    _, num_columns = X_train.shape

    for column_index in range(0, num_columns):
        column = X_train[:, column_index]
        if not _is_binary(column):
            if not _has_zero_mean(column):
                print("Column", column_index, "does not have zero mean")
                return True
            if not _has_unit_variance(column):
                print("Column", column_index, "does not have unit variance")
                return True

    return False
