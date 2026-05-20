import sys
import logging
import numpy as np
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

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
        run_id, event_id, energy = self.meta[index]
        # Create image
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        fig.suptitle(f'Fermi-LAT All Views (Run {run_id:.0f}, Event {event_id:.0f}, {energy:.2f} MeV)', fontsize=16)
        # X-Z View
        axes[0].imshow(self.view_x[index], cmap='plasma', origin='lower', vmin=0, vmax=1)
        axes[0].set_title('X-Z Projection (Side View)')
        axes[0].set_xlabel('LAT X Width (113 Pixels)')
        axes[0].set_ylabel('LAT Z Height (113 Pixels)')
        # axes[0].legend(loc='upper right')
        # Y-Z View
        axes[1].imshow(self.view_y[index], cmap='plasma', origin='lower', vmin=0, vmax=1)
        axes[1].set_title('Y-Z Projection (Side View)')
        axes[1].set_xlabel('LAT Y Width (113 Pixels)')
        axes[1].set_ylabel('LAT Z Height (113 Pixels)')
        # X-Y Top View
        mesh_top = axes[2].imshow(self.view_top[index], cmap='plasma', origin='lower', vmin=0, vmax=1)
        axes[2].set_title('X-Y Projection (CAL Top-Down)')
        axes[2].set_xlabel('LAT X Width (113 Pixels)')
        axes[2].set_ylabel('LAT Y Width (113 Pixels)')
        # Colorbar
        fig.colorbar(mesh_top, ax=axes, label='Normalized Log10(Energy)', shrink=0.8, pad=0.02)
        # Plot
        plt.show()

    def close(self) -> None:
        """ Closes the numpy archive to free up memory.
        """
        self.archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and Visualize compressed LAT chunks.')
    parser.add_argument('file', type=str, help='Path to the .npz chunk file to validate')
    parser.add_argument('--event', type=int, default=0, help='Index of the event to visualize (default: 0)')
    parser.add_argument('--no-plot', action='store_true', help='Skip the visualization and only run validation checks')

    args = parser.parse_args()
    validator = ChunkValidator(args.file)
    # Run the validation checks
    validator.verify_shapes()
    validator.print_summary()
    # Optionally visualize
    if not args.no_plot:
        logger.info(f"Rendering visualization for event index {args.event}...")
        validator.visualize_event(index=args.event)
    # Close
    validator.close()