import sys

import numpy as np
import pandas as pd

from .event import Event
from utils import normalize_log_matrix


class CalData:
    """ Class that handles calorimeter data.
    """

    SIDE    = 1563.6/2      # mm
    HEIGHT  = -777          # mm
    NUM_BINS_SIDE = 48*2
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

    def get_matrix_side(self, coord: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Returns the matrix and the binning for a side projection.
        """
        if coord.lower() == 'x':
            coord_data = self.x
        elif coord.lower() == 'y':
            coord_data = self.y
        else:
            sys.exit("Please choose 'x' or 'y' as tracker coordinate.")
        matrix, x1_edges, z_edges = np.histogram2d(coord_data, self.z,
                                                   bins=[int(self.SIDE*2/self.event.BIN_WIDTH), self.NUM_BINS_Z],
                                                   range=[[-self.SIDE,self.SIDE], [self.z.min(),self.z.max()]],
                                                   weights=self.E)
        matrix_masked:np.ndarray    = np.ma.masked_where(matrix == 0, matrix)
        norm_matrix: np.ndarray     = normalize_log_matrix(matrix_masked, norm=self.event.total_energy)
        return norm_matrix.T, x1_edges, z_edges
    
    def get_matrix_top(self):
        """ Returns the matrix and the binning for the top projection.
        """
        matrix, x_edges, y_edges = np.histogram2d(self.x, self.y,
                                                   bins=[self.NUM_BINS_SIDE, self.NUM_BINS_SIDE],
                                                   range=[[-self.SIDE,self.SIDE], [-self.SIDE,self.SIDE]],
                                                   weights=self.E)
        matrix_masked:np.ndarray    = np.ma.masked_where(matrix == 0, matrix)
        norm_matrix: np.ndarray     = normalize_log_matrix(matrix=matrix_masked, norm=self.event.total_energy)
        return norm_matrix.T, x_edges, y_edges
