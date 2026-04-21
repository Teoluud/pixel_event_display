import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


class EventDisplay:
    """ 
    Dedicated class to handle all Matplotlib rendering.
    """

    def __init__(self, run_id: str, event_id: str, event_energy: float) -> None:
        """ Constructor.
        """
        self.run_id = run_id
        self.event_id = event_id
        self.event_energy = event_energy

    def plot_single_view(self, x_edges: np.ndarray, z_edges: np.ndarray, matrix: np.ndarray, view_name: str, detector: str = 'TKR') -> tuple[Figure, Axes]:
        """ 
        Plots a single 2D projection (ex.: TKR X-Z or CAL Y-Z).
        Assumes a normalized matrix.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        mesh = ax.pcolormesh(x_edges, z_edges, matrix, cmap='plasma', shading='flat', vmin=0, vmax=1)
        fig.colorbar(mesh, ax=ax, label='Normalized Log10(Energy)')
        ax.set_xlabel(f'{view_name.upper()} [mm]')
        ax.set_ylabel(f'Z [mm]')
        ax.set_title(f'Fermi-LAT {detector}, {view_name.upper()}-Z view (Run {self.run_id}, Event {self.event_id})')
        return fig, ax
    
    def plot_combined_view(self, tkr_x: np.ndarray, tkr_z: np.ndarray, tkr_matrix: np.ndarray, 
                           cal_x: np.ndarray, cal_z: np.ndarray, cal_matrix: np.ndarray, view_name: str = 'x') -> tuple[Figure, Axes]:
        """ Overlays TKR and CAL matrices on the same plot for the full representation.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        # Draw CAL first (figure background), then TKR.
        ax.pcolormesh(cal_x, cal_z, cal_matrix, cmap='plasma', shading='flat', vmin=0, vmax=1)
        mesh = ax.pcolormesh(tkr_x, tkr_z, tkr_matrix, cmap='plasma', shading='flat', vmin=0, vmax=1)
        fig.colorbar(mesh, ax=ax, label='Normalized Log10(Energy)')
        ax.set_xlabel(f'{view_name.upper()} [mm]')
        ax.set_ylabel(f'Z [mm]')
        ax.set_title(f'Fermi-LAT Full {view_name.upper()}-Z view (Run {self.run_id}, Event {self.event_id}, Energy {self.event_energy:.3e} MeV)')
        return fig, ax
    
    def plot_topdown_view(self, cal_x: np.ndarray, cal_y: np.ndarray, cal_matrix: np.ndarray) -> tuple[Figure, Axes]:
        """ Plots the top-down 2D projection of the calorimeter (CAL X-Y).
        """
        fig, ax = plt.subplots(figsize=(10,6))
        mesh = ax.pcolormesh(cal_x, cal_y, cal_matrix, cmap='plasma', shading='flat', vmin=0, vmax=1)
        fig.colorbar(mesh, ax=ax, label='Normalized Log10(Energy)')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title(f'Fermi-LAT X-Y view (Run {self.run_id}, Event {self.event_id}, Energy {self.event_energy:.3e} MeV)')
        return fig, ax
    
    def show_all(self):
        """
        Triggers the display of all generated figures.
        """
        plt.show()