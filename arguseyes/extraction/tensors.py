import numpy as np


# Create a matrix from wrapped numpy arrays (workaround, will be fixed later)
def copy_to_matrix(wrapped):
    matrix = np.zeros((len(wrapped), len(wrapped[0].flatten())), dtype='float64')

    for i in range(0, len(wrapped)):
        matrix[i, :] = wrapped[i].flatten()[:]

    return matrix
