import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .event import Event
from .utils import get_lat_x_edges


class TkrData:
    """ Class that handles the tracker data processing.
    """
    NUM_BINS_SIDE = 48*2
    NUM_BINS_Z = 18
    # To better represent real LAT dimensions, use BIN_HEIGHT and BIN_WIDTH. (from class Event)

    def __init__(self, dataframe: pd.DataFrame, event: Event, config: dict[str, Any]) -> None:
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
        self.x: NDArray[np.float32]  = np.asarray(self.df_x['Coord'], dtype=np.float32)
        self.y: NDArray[np.float32]  = np.asarray(self.df_y['Coord'], dtype=np.float32)
        self.z: NDArray[np.float32]  = np.asarray(df['Z'], dtype=np.float32)
        self.tot_x: NDArray[np.float32] = np.asarray(self.df_x['ToT'], dtype=np.float32)
        self.tot_y: NDArray[np.float32] = np.asarray(self.df_y['ToT'], dtype=np.float32)

    def get_matrix(self, coord: str) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """ Returns the matrix and the binning.
        """
        if coord.lower() == 'x':
            coord_data  = self.x
            z_data      = np.asarray(self.df_x['Z'], dtype = np.float32)
            tot_data    = self.tot_x
            custom_z_edges = self.config['geometry']['tkr']['z_edges_x']
        elif coord.lower() == 'y':
            coord_data  = self.y
            z_data      = np.asarray(self.df_y['Z'], dtype=np.float32)
            tot_data    = self.tot_y
            custom_z_edges = self.config['geometry']['tkr']['z_edges_y']
        else:
            sys.exit("Please choose 'x' or 'y' as tracker coordinate.")
        self.E = tot_data / 1.602 * 3.62e-2
        custom_x_edges = get_lat_x_edges(self.config)
        matrix, x1_edges, z_edges = np.histogram2d(coord_data, z_data, 
                                            bins=[custom_x_edges, custom_z_edges], 
                                            #range=[[-self.SIDE,self.SIDE], [0., self.HEIGHT]],
                                            weights=self.E)
        matrix_masked: NDArray[np.float32]   = np.ma.masked_where(matrix <= 0, matrix)
        return matrix_masked.T, x1_edges, z_edges