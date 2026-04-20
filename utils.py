import numpy as np


def normalize_log_matrix(matrix: np.ndarray, norm: float) -> np.ndarray:
    """ Normalizes the log matrix to the norm value.
    """
    log_matrix  = np.ma.log10(matrix)
    norm_matrix = log_matrix / np.log10(norm)
    return norm_matrix