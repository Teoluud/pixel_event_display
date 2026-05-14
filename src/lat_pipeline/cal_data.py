import sys

import numpy as np
import pandas as pd

from .event import Event


class CalData:
    """ Class that handles calorimeter data.
    """
    NUM_BINS_Z = 8
    
    def __init__(self, dataframe: pd.DataFrame, event: Event, config: dict) -> None:
        """ Constructor.
        """
        self.event = event
        self.config = config
        # Load geometry from config
        self.SIDE = self.config['geometry']['side_mm']
        self.HEIGHT = self.config['geometry']['cal']['height_mm']
        self.PITCH = self.config['geometry']['cal']['pitch_mm']
        self.TOWER_GAP = self.config['geometry']['cal']['tower_gap_mm']
        self.EXT_GAP = self.config['geometry']['cal']['ext_gap_mm']
        # Extract data
        self.df: pd.DataFrame   = dataframe
        raw_x = self.df['X'].to_numpy()
        raw_y = self.df['Y'].to_numpy()
        # Apply micro-jitter to all hits
        self.x: np.ndarray      = raw_x + np.random.uniform(-1e-5, 1e-5, size=len(raw_x))
        self.y: np.ndarray      = raw_y + np.random.uniform(-1e-5, 1e-5, size=len(raw_y))
        self.z: np.ndarray      = self.df['Z'].to_numpy()
        self.E: np.ndarray      = self.df['Energy_MeV'].to_numpy()

    def generate_horizontal_edges(self, start_coord: float, crystals_per_tower: int = 12, num_towers: int = 4) -> np.ndarray:
        """ 
        Generates the horizontal pixel boundaries:
        - 2 bins per crystal
        - 3 bins per tower gap
        - 4 bins for the outer gaps
        """
        edges = [start_coord]
        current_pos = start_coord
        # Add the pixels for the left outer gap
        step = self.EXT_GAP / 4.0
        for t in range(4):
            current_pos += step
            edges.append(current_pos)
        # Add the pixels for the CAL
        for tower in range(num_towers):
            for c in range(crystals_per_tower):
                half_pitch = self.PITCH / 2.0
                current_pos += half_pitch
                edges.append(current_pos)
                current_pos += half_pitch
                edges.append(current_pos)
            # Bins for the gaps between towers
            if tower < num_towers -1:
                step = self.TOWER_GAP / 3.0
                for g in range(3):
                    current_pos += step
                    edges.append(current_pos)
        # Add the pixels for the right outer gap
        step = self.EXT_GAP / 4.0
        for t in range(4):
            current_pos += step
            edges.append(current_pos)
        return np.array(edges)
    
    def split_boundary_hits(self, coord_data: np.ndarray, bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes the hits on the middle of the crystal (which then are right on the bin boundary), and split them on the adjacient bins.
        It doubles the hit, then halves the energy and assigns one to the left of the boundary and one to the right.

        (NOT IN USE)
        """
        # Identify which hits fall on the bin boundary
        # We initiate the mask here as all 0
        on_edge_mask = np.zeros(len(coord_data), dtype=bool)
        # Check if there are hits on the edge (using np.isclose to avoid rounding errors)
        for edge in bin_edges[1:-1]:
            # Update the mask adding the condition to be close to the bin edge
            on_edge_mask |= np.isclose(coord_data, edge, atol=1e-5)
        # Separate hits not on edge
        normal_coord = coord_data[~on_edge_mask]                       # ~ is the bitwise NOT operator
        normal_z = self.z[~on_edge_mask]
        normal_E = self.E[~on_edge_mask]
        # Separate hits on the edge
        edge_coord = coord_data[on_edge_mask]
        edge_z = self.z[on_edge_mask]
        edge_E = self.E[on_edge_mask] / 2.0
        # Create the left and right copies
        shift = 1e-4
        left_coord = edge_coord - shift
        right_coord = edge_coord + shift
        # Recombine the arrays
        final_coord = np.concatenate([normal_coord, left_coord, right_coord])
        final_z = np.concatenate([normal_z, edge_z, edge_z])
        final_E = np.concatenate([normal_E, edge_E, edge_E])        
        return final_coord, final_z, final_E


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
        matrix, x1_edges, z_edges = np.histogram2d(coord_data, self.z,
                                                   bins=[custom_x1_edges, int(self.HEIGHT/self.event.BIN_HEIGHT)],
                                                   #range=[[-self.SIDE,self.SIDE], [self.z.min(),self.z.max()]],
                                                   weights=self.E)
        matrix_masked:np.ndarray = np.ma.masked_where(matrix == 0, matrix)
        return matrix_masked.T, x1_edges, z_edges
    
    def get_matrix_top(self):
        """ Returns the matrix and the binning for the top projection.
        """
        x_data = self.x.copy()
        y_data = self.y.copy()
        custom_edges = self.generate_horizontal_edges(start_coord=-self.SIDE)
        matrix, x_edges, y_edges = np.histogram2d(x_data, y_data,
                                                   bins=[custom_edges, custom_edges],
                                                   #range=[[-self.SIDE,self.SIDE], [-self.SIDE,self.SIDE]],
                                                   weights=self.E)
        matrix_masked:np.ndarray = np.ma.masked_where(matrix == 0, matrix)
        return matrix_masked.T, x_edges, y_edges
