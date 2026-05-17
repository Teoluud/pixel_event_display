import numpy as np


def normalize_log_matrix(matrix: np.ndarray, norm: float) -> np.ndarray:
    """ Calculates the log matrix and returns it normalized to the norm value.
    """
    matrix_kev   = matrix * 1e3
    norm_kev     = norm * 1e3
    log_matrix  = np.ma.log10(matrix_kev)
    norm_matrix = log_matrix / np.log10(norm_kev)
    return norm_matrix


def get_lat_x_edges(config: dict) -> np.ndarray:
    """ Generates the exact 114 edges (113 bins) for the LAT X/Y dimensions.
    """
    # Load geometry from config
    side_mm = config['geometry']['side_mm']
    pitch = config['geometry']['cal']['pitch_mm']
    tower_gap = config['geometry']['cal']['tower_gap_mm']
    ext_gap = config['geometry']['cal']['ext_gap_mm']
    # CREATE CUSTOM EDGES
    edges = [-side_mm]
    current_pos = -side_mm
    # Left outer gap (4 pixels)
    step = ext_gap / 4.0
    for _ in range(4):
        current_pos += step
        edges.append(current_pos)
    # Towers
    for tower in range(4):
        for _ in range(12):
            half_pitch = pitch / 2.0
            current_pos += half_pitch
            edges.append(current_pos)
            current_pos += half_pitch
            edges.append(current_pos)
        # Gaps between towers
        if tower < 3:
            step = tower_gap / 3.0
            for _ in range(3):
                current_pos += step
                edges.append(current_pos)
    # Right outer gap (4 pixels)
    step = ext_gap / 4.0
    for _ in range(4):
        current_pos += step
        edges.append(current_pos)
    return np.array(edges)