import sys
import logging
import numpy as np
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunkValidator:
    """ A diagnostic tool to load, verify and visualize FERMI-LAT chunks.
    """

    def __init__(self, filepath: str | Path) -> None:
        """ Constructor.
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            logger.error(f'File not found: {self.filepath}')
            sys.exit(1)
        try:
            self.archive = np.load(self.filepath)
            self.view_x: np.ndarray = self.archive['view_x']
            self.view_y: np.ndarray = self.archive['view_y']
            self.view_top: np.ndarray = self.archive['view_top']
            self.meta: np.ndarray = self.archive['meta']
            self.num_events = self.meta.shape[0]
            logger.info(f'Successfully loaded {self.filepath.name} containing {self.num_events} events.')
        except KeyError as e:
            logger.error(f'Missing expected tensor key in archive: {e}')
            sys.exit(1)

    def verify_shapes(self) -> bool:
        """ Verify that the tensors match the expected geometry.
        """
        expected_side = (113, 113)
        expected_top = (113, 113)
        is_valid = True
        if self.view_x.shape[1:] != expected_side:
            logger.error(f'View X shape mismatch: got {self.view_x.shape[1:]}, expected {expected_side}')
            is_valid = False
        if self.view_y.shape[1:] != expected_side:
            logger.error(f'View Y shape mismatch: got {self.view_y.shape[1:]}, expected {expected_side}')
            is_valid = False
        if self.view_top.shape[1:] != expected_top:
            logger.error(f'View Top shape mismatch: got {self.view_top.shape[1:]}, expected {expected_top}')
        if is_valid:
            logger.info('SUCCESS: All tensor dimensions match the expected geometry.')
        return is_valid
    
    def print_summary(self) -> None:
        """ Prints statistical sanity checks on the dataset.
        """
        logger.info('--- Data Summary ---')
        logger.info(f'Total Events: {self.num_events}')
        # Calculate global max and min across all views
        global_max = max(self.view_x.max(), self.view_y.max(), self.view_top.max())
        global_min = min(self.view_x.min(), self.view_y.min(), self.view_top.min())
        logger.info(f'Global Max Pixel Value: {global_max:.4f}')
        logger.info(f'Global Min Pixel Value: {global_min:.4f}')
        # Check all views for NaN or Inf values
        has_nans = np.isnan(self.view_x).any() or np.isnan(self.view_y).any() or np.isnan(self.view_top).any()
        has_infs = np.isinf(self.view_x).any() or np.isinf(self.view_y).any() or np.isinf(self.view_top).any()
        if has_nans or has_infs:
            logger.error('Found NaN or Infinite values in the data!')
        else:
            logger.info('Data Integrity: No NaN or Infinite values detected.')
    
    def visualize_event(self, index: int = 0) -> None:
        """ Renders all the projections for a specific event index.
        """
        if index >= self.num_events or index < 0:
            logger.error(f'Index {index} is out of bounds. File only has {self.num_events} events.')
            return
        run_id, event_id, energy, mc_energy = self.meta[index]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        fig.suptitle(f'Fermi-LAT All Views (Run {run_id:.0f}, Event {event_id:.0f}, Energy {energy:.2f} MeV, McEnergy {mc_energy:.2f})', fontsize=16)
        
        # Determine dynamic maxes for the log scale (fallback to 1.0 if empty)
        vmax_x = self.view_x[index].max() if self.view_x[index].max() > 0 else 1.0
        vmax_y = self.view_y[index].max() if self.view_y[index].max() > 0 else 1.0
        vmax_top = self.view_top[index].max() if self.view_top[index].max() > 0 else 1.0

        # --- X-Z View ---
        axes[0].imshow(self.view_x[index], cmap='plasma', origin='lower', norm=LogNorm(vmin=1e-3, vmax=vmax_x + 1))
        axes[0].set_title('X-Z Projection (Side View)')
        axes[0].set_xlabel('LAT X Width (113 Pixels)')
        axes[0].set_ylabel('LAT Z Height (113 Pixels)')
        
        # --- Y-Z View ---
        axes[1].imshow(self.view_y[index], cmap='plasma', origin='lower', norm=LogNorm(vmin=1e-3, vmax=vmax_y + 1))
        axes[1].set_title('Y-Z Projection (Side View)')
        axes[1].set_xlabel('LAT Y Width (113 Pixels)')
        axes[1].set_ylabel('LAT Z Height (113 Pixels)')
        
        # --- X-Y Top View ---
        mesh_top = axes[2].imshow(self.view_top[index], cmap='plasma', origin='lower', norm=LogNorm(vmin=1e-3, vmax=vmax_top + 1))
        axes[2].set_title('X-Y Projection (CAL Top-Down)')
        axes[2].set_xlabel('LAT X Width (113 Pixels)')
        axes[2].set_ylabel('LAT Y Width (113 Pixels)')
        
        # Colorbar that shows raw MeVs
        fig.colorbar(mesh_top, ax=axes, label='Energy [MeV] (Log Scale)', shrink=0.8, pad=0.02)

    def check_energy_recon(self) -> None:
        """ Plots the distribution of the energy reconstruction error.
        """
        energy = self.meta[:, 2]
        mc_energy = self.meta[:, 3]
        err_energy_relative = (energy - mc_energy) / mc_energy
        bin_size = 0.1
        num_bins = int((err_energy_relative.max() - err_energy_relative.min()) / bin_size)
        plt.figure(figsize=(10,6))
        plt.hist(err_energy_relative, bins=num_bins)
        plt.xlabel("Reconstructed energy relative error")
        plt.title(f"Chunk: {self.filepath.name}")

    def energy_spectrum(self) -> None:
        """ Plots the reconstructed energy distribution.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
        fig.suptitle(f"Energy spectrum comparison - Chunk: {self.filepath.name}")
        num_bins = 50
        energy = self.meta[:, 2]
        
        # Safely create bin edges evenly spaced in log-space (skip 0s)
        safe_energy = energy[energy > 0]
        if len(safe_energy) > 0:
            min_log = np.log10(safe_energy.min())
            max_log = np.log10(safe_energy.max())
            log_bins = np.logspace(min_log, max_log, num_bins)
            axes[0].hist(safe_energy, bins=log_bins, edgecolor='black', alpha=0.8)
            axes[0].set_xscale("log")
        axes[0].set_xlabel("EvtJointEnergy [MeV]")
        axes[0].set_title("Reconstructed Energy")
        
        # MC energy spectrum
        mc_energy = self.meta[:, 3]
        safe_mc = mc_energy[mc_energy > 0]
        if len(safe_mc) > 0:
            min_log = np.log10(safe_mc.min())
            max_log = np.log10(safe_mc.max())
            log_bins = np.logspace(min_log, max_log, num_bins)
            axes[1].hist(safe_mc, bins=log_bins, edgecolor='black', alpha=0.8)
            axes[1].set_xscale("log")
        axes[1].set_xlabel("McEnergy [MeV]")
        axes[1].set_title("Simulated Energy")

    def plot(self) -> None:
        plt.show()

    def close(self) -> None:
        """ Closes the numpy archive to free up memory.
        """
        self.archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and Visualize compressed LAT chunks.')
    parser.add_argument('file', type=str, help='Path to the .npz chunk file to validate')
    
    # --- Analysis Arguments ---
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--check-energy', action='store_true', help='Plot the distribution of the energy reconstruction error.')
    analysis_group.add_argument('--energy-spectrum', action='store_true', help='Plot the reconstructed energy distribution.')
    
    # --- Visualization Arguments ---
    vis_group = parser.add_argument_group('Visualization Options')
    vis_group.add_argument('--event', type=int, default=None, help='Index of a specific event to visualize')
    vis_group.add_argument('--loop', action='store_true', help='Loop through all events sequentially')
    vis_group.add_argument('--no-plot', action='store_true', help='Skip all visualizations and only run validation checks')

    args = parser.parse_args()
    validator = ChunkValidator(args.file)
    
    # Run the Validation Checks
    validator.verify_shapes()
    validator.print_summary()
    
    # Exit early if the user just wanted a silent data check
    if args.no_plot:
        validator.close()
        sys.exit(0)
        
    # Handle Global Analysis Plots (Energy Spectra, etc.)
    show_global_plots = False
    if args.check_energy:
        validator.check_energy_recon()
        show_global_plots = True
        
    if args.energy_spectrum:
        validator.energy_spectrum()
        show_global_plots = True
        
    if show_global_plots:
        logger.info("Displaying global analysis plots. Close the plot windows to continue.")
        validator.plot()
        
    # Handle Event Visualization (Loop or Single Event)
    if args.loop:
        logger.info("Entering event loop mode...")
        for i in range(validator.num_events):
            logger.info(f"Rendering event {i} / {validator.num_events - 1}...")
            validator.visualize_event(index=i)
            validator.plot()
            
            # The script pauses here while the window is open. 
            # Once closed, ask the user how to proceed:
            try:
                ans = input(f"Event {i} closed. Press [Enter] for next event, or type 'q' to quit: ")
                if ans.strip().lower() == 'q':
                    logger.info("Exiting event loop.")
                    break
            except KeyboardInterrupt:
                break
                
    elif args.event is not None:
        # Show specific event if requested
        logger.info(f"Rendering visualization for requested event index {args.event}...")
        validator.visualize_event(index=args.event)
        validator.plot()
        
    elif not show_global_plots:
        # Default behavior: If they didn't ask for any plots specifically, just show event 0
        logger.info("Rendering visualization for default event index 0...")
        validator.visualize_event(index=0)
        validator.plot()

    # Clean up memory
    validator.close()