import numpy as np


def _is_constant(column):
    unique_values = np.unique(column)
    return len(unique_values) == 1


def detect(classification_pipeline) -> bool:

    X_train = classification_pipeline.X_train
    _, num_columns = X_train.shape

    constant_column_found = False

    for column_index in range(0, num_columns):
        column = X_train[:, column_index]
        if _is_constant(column):
            print("Column", column_index, "is constant")
            constant_column_found = True

    return constant_column_found
