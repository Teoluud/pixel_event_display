import sys

import numpy as np
import pandas as pd

from .event import Event


class TkrData:
    """ Class that handles the tracker data processing.
    """
    NUM_BINS_SIDE = 48*2
    NUM_BINS_Z = 18
    # To better represent real LAT dimensions, use BIN_HEIGHT and BIN_WIDTH. (from class Event)

    def __init__(self, dataframe: pd.DataFrame, event: Event, config: dict) -> None:
        """ Constructor.
        """
        self.event = event
        self.config = config
        # Load geometry from config
        self.SIDE = self.config['geometry']['side_mm']
        self.HEIGHT = self.config['geometry']['tkr']['height_mm']
        # Extract data
        df: pd.DataFrame        = dataframe
        self.df_x: pd.DataFrame = df[df['View'] == 'X']
        self.df_y: pd.DataFrame = df[df['View'] == 'Y']
        self.x: np.ndarray  = self.df_x['Coord'].to_numpy()
        self.y: np.ndarray  = self.df_y['Coord'].to_numpy()
        self.z: np.ndarray  = df['Z'].to_numpy()
        self.tot_x: np.ndarray = self.df_x['ToT'].to_numpy()
        self.tot_y: np.ndarray = self.df_y['ToT'].to_numpy()

    def get_matrix(self, coord: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Returns the matrix and the binning.
        """
        if coord.lower() == 'x':
            coord_data  = self.x
            z_data      = self.df_x['Z'].to_numpy()
            tot_data    = self.tot_x
        elif coord.lower() == 'y':
            coord_data  = self.y
            z_data      = self.df_y['Z'].to_numpy()
            tot_data    = self.tot_y
        else:
            sys.exit("Please choose 'x' or 'y' as tracker coordinate.")
        self.E = tot_data / 1.602 * 3.62e-2
        matrix, x1_edges, z_edges = np.histogram2d(coord_data, z_data, 
                                            bins=[int(self.SIDE*2/self.event.BIN_WIDTH), int(self.HEIGHT/self.event.BIN_HEIGHT)], 
                                            range=[[-self.SIDE,self.SIDE], [0., self.HEIGHT]],
                                            weights=self.E)
        matrix_masked: np.ndarray   = np.ma.masked_where(matrix <= 0, matrix)
        return matrix_masked.T, x1_edges, z_edges