
import math
import numpy as np


time = 0.1  # s
image = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

T1_matrix = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])

T2_matrix = np.array([[0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1]])


def decay_recovery_matrix(time, row_index, col_index):
    T1 = T1_matrix[row_index, col_index]
    T2 = T2_matrix[row_index, col_index]
    decay_recovery_matrix = np.array([[math.exp(-time/T2), 0, 0],
                                      [0, math.exp(-time/T2), 0],
                                      [0, 0, 1 - math.exp(-time/T1)]])

    return decay_recovery_matrix
