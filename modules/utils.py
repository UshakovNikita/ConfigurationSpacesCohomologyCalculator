import numpy as np


class PosetValidationError(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return f'Poset is not valid:  {self.error}'


def permutation_matrix(elems: list) -> np.ndarray:
    size = len(elems)
    matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        matrix[elems[i], i] = 1
    return matrix


def all_ones_matrix(row, col: int) -> np.ndarray:
    return np.ones((row, col), dtype=int)



