import numpy as np


def normalize_log_matrix(matrix: np.ndarray, norm: float) -> np.ndarray:
    """ Calculates the log matrix and returns it normalized to the norm value.
    """
    matrix_kev   = matrix * 1e3
    norm_kev     = norm * 1e3
    log_matrix  = np.ma.log10(matrix_kev)
    norm_matrix = log_matrix / np.log10(norm_kev)
    return norm_matrix