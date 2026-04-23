import sys

import numpy as np
import pandas as pd

from .event import Event
from utils import normalize_log_matrix


class CalData:
    """ Class that handles calorimeter data.
    """

    SIDE    = 1563.6/2      # mm
    HEIGHT  = 218.2         # mm
    PITCH   = 27.84         # mm
    TOWER_GAP = 40.42       # mm
    EXT_GAP   = 53.01       # mm
    NUM_BINS_Z = 8
    
    def __init__(self, dataframe: pd.DataFrame, event: Event) -> None:
        """ Constructor.
        """
        self.event = event
        self.df: pd.DataFrame   = dataframe
        self.x: np.ndarray      = self.df['X'].to_numpy()
        self.y: np.ndarray      = self.df['Y'].to_numpy()
        self.z: np.ndarray      = self.df['Z'].to_numpy()
        self.E: np.ndarray      = self.df['Energy_MeV'].to_numpy()

    def generate_horizontal_edges(self, start_coord: float, crystal_pitch: float = PITCH, gap_width: float = TOWER_GAP, external_gap: float = EXT_GAP,
                                  crystals_per_tower: int = 12, num_towers: int = 4) -> np.ndarray:
        """ 
        Generates the horizontal pixel boundaries:
        - 2 bins per crystal
        - 3 bins per tower gap
        """
        edges = [start_coord]
        current_pos = start_coord
        # Add the pixels for the left outer gap
        step = external_gap / 4.0
        for t in range(4):
            current_pos += step
            edges.append(current_pos)
        # Add the pixels for the CAL
        for tower in range(num_towers):
            for c in range(crystals_per_tower):
                half_pitch = crystal_pitch / 2.0
                current_pos += half_pitch
                edges.append(current_pos)
                current_pos += half_pitch
                edges.append(current_pos)
            # Bins for the gaps between towers
            if tower < num_towers -1:
                step = gap_width / 3.0
                for g in range(3):
                    current_pos += step
                    edges.append(current_pos)
        # Add the pixels for the right outer gap
        step = external_gap / 4.0
        for t in range(4):
            current_pos += step
            edges.append(current_pos)
        return np.array(edges)


    def get_matrix_side(self, coord: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Returns the matrix and the binning for a side projection.
        """
        if coord.lower() == 'x':
            coord_data = self.x.copy()      # We use copy in order not to alter the original data when we add artificial noise.
        elif coord.lower() == 'y':
            coord_data = self.y.copy()
        else:
            sys.exit("Please choose 'x' or 'y' as tracker coordinate.")
        # Manually set the binning to better represent the CAL dimensions, without breaking the pixels.
        custom_x1_edges = self.generate_horizontal_edges(start_coord=-self.SIDE)
        breakpoint()
        # Add micro-jitter to force 50/50 split on exact crystal boundaries
        jitter = np.random.uniform(low=-1e-5, high=1e-5, size=len(coord_data))
        coord_data += jitter
        matrix, x1_edges, z_edges = np.histogram2d(coord_data, self.z,
                                                   bins=[custom_x1_edges, int(self.HEIGHT/self.event.BIN_HEIGHT)],
                                                   #range=[[-self.SIDE,self.SIDE], [self.z.min(),self.z.max()]],
                                                   weights=self.E)
        matrix_masked:np.ndarray    = np.ma.masked_where(matrix == 0, matrix)
        return matrix_masked.T, x1_edges, z_edges
    
    def get_matrix_top(self):
        """ Returns the matrix and the binning for the top projection.
        """
        x_data = self.x.copy()
        y_data = self.y.copy()
        custom_edges = self.generate_horizontal_edges(start_coord=-self.SIDE)
        jitter = np.random.uniform(low=-1e-5, high=1e-5, size=len(x_data))
        x_data += jitter
        y_data += jitter
        matrix, x_edges, y_edges = np.histogram2d(x_data, y_data,
                                                   bins=[custom_edges, custom_edges],
                                                   #range=[[-self.SIDE,self.SIDE], [-self.SIDE,self.SIDE]],
                                                   weights=self.E)
        matrix_masked:np.ndarray    = np.ma.masked_where(matrix == 0, matrix)
        return matrix_masked.T, x_edges, y_edges
