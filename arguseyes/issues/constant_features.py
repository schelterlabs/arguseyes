import numpy as np


def _is_constant(column):
    unique_values = np.unique(column)
    return len(unique_values) == 1


# TODO for the bias term in linear models, its ok to be constant, how to handle this?
def detect(classification_pipeline) -> bool:

    X_train = classification_pipeline.X_train
    _, num_columns = X_train.shape

    for column_index in range(0, num_columns):
        column = X_train[:, column_index]
        if _is_constant(column):
            print("Column", column_index, "is constant")
            return True

    return False
