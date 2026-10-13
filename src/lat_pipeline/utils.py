from typing import Any

import numpy as np
from numpy.typing import NDArray


def get_lat_x_edges(config: dict[str, Any]) -> NDArray[np.int64]:
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