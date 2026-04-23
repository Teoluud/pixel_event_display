import numpy as np


def normalize_log_matrix(matrix: np.ndarray, norm: float) -> np.ndarray:
    """ Calculates the log matrix and returns it normalized to the norm value.
    """
    log_matrix  = np.ma.log10(matrix)
    norm_matrix = log_matrix / np.log10(norm)
    return norm_matrix