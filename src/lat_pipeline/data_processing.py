import numpy as np

from .stream_parser import StreamParser
from .import_data import ImportData
from .event_display import EventDisplay
from .utils import normalize_log_matrix


class DataProcessing:
    """ 
    High-level orchestrator class. 
    It commands the ImportData (Controller), fetches processed matrices, and pushes them to the EventDisplay (View).
    """
    def __init__(self, path: str, run_id: str, event_id: str, config: dict) -> None:
        """
        Constructor.
        """
        self.path       = path
        self.run_id     = run_id
        self.event_id   = event_id
        self.config     = config

    def process_and_display(self, view: str = 'x') -> None:
        """
        Runs the full pipeline for a specific projection.
        """
        # Import data
        importer = ImportData(self.path, self.run_id, self.event_id, config=self.config)
        event, tkr, cal = importer.import_event()
        # Instantiate the display view
        self.display = EventDisplay(self.run_id, self.event_id, event_energy=event.total_energy)
        # Extract matrices and render
        if view.lower() == 'top':
            cal_matrix_top, cal_x, cal_y    = cal.get_matrix_top()
            norm_cal_matrix_top = normalize_log_matrix(cal_matrix_top, norm=event.total_energy)
            self.display.plot_topdown_view(cal_x, cal_y, norm_cal_matrix_top)
        else:
            tkr_matrix, tkr_x, tkr_z        = tkr.get_matrix(view)
            norm_tkr_matrix = normalize_log_matrix(tkr_matrix, norm=event.total_energy)
            cal_matrix_side, cal_x, cal_z   = cal.get_matrix_side(view)
            norm_cal_matrix_side = normalize_log_matrix(cal_matrix_side, norm=event.total_energy)
            # Render via the View
            self.display.plot_combined_view(tkr_x, tkr_z, norm_tkr_matrix, cal_x, cal_z, norm_cal_matrix_side, view_name=view)
            # self.display.plot_single_view(tkr_x, tkr_z, tkr_matrix, view_name=view, detector='TKR')
            # self.display.plot_single_view(cal_x, cal_z, cal_matrix_side, view_name=view, detector='CAL')
        self.display.show_all()


class BatchProcessor:
    """ Processes piped events and saves matrices in bulk.
    """

    def __init__(self, config: dict) -> None:
        """ Constructor.
        """
        self.config = config
        self.chunk_size = self.config['pipeline']['chunk_size']
        self.stream_parser = StreamParser(config=self.config)

    def process_and_save(self, view: str = 'x', output_prefix: str = 'dataset') -> None:
        """ Runs the full pipeline for a specific projection in a batch.
        """
        tkr_matrices = []
        cal_matrices = []
        event_infos = []    # Store run_id, event_id, energy as metadata
        chunk_index = 0
        event_count = 0
        # Iterate through the piped stream dinamically
        for event, tkr, cal in self.stream_parser.parse_stream():
            # Extract matrices
            tkr_matrix, _, _ = tkr.get_matrix(view)
            cal_matrix, _, _ = cal.get_matrix_side(view)
            # Normalize to keV
            norm_tkr = normalize_log_matrix(tkr_matrix, norm=event.total_energy)
            norm_cal = normalize_log_matrix(cal_matrix, norm=event.total_energy)
            # Fill masked areas with 0 for the NN
            tkr_matrices.append(np.ma.filled(norm_tkr, fill_value=0.0))
            cal_matrices.append(np.ma.filled(norm_cal, fill_value=0.0))
            event_infos.append([event.run_id, event.event_id, event.total_energy])
            event_count += 1
            # Save and flush memory when chunk is full
            if event_count >= self.chunk_size:
                self._save_chunk(tkr_matrices, cal_matrices, event_infos, output_prefix, chunk_index)
                # Reset lists to free up RAM
                tkr_matrices = []
                cal_matrices = []
                event_infos = []
                chunk_index += 1
                event_count = 0
        # Save any remaining events after the pipe closes
        if event_count > 0:
            self._save_chunk(tkr_matrices, cal_matrices, event_infos, output_prefix, chunk_index)

    def _save_chunk(self, tkr, cal, info, prefix, index):
        """ Helper to save a list of matrices to a compressed numpy archive.
        """
        filename = f'{prefix}_chunk_{index:04d}.npz'
        # Convert lists to 3D numpy arrays: shape (N_events, Height, Width)
        tkr_arr = np.array(tkr, dtype=np.float32)
        cal_arr = np.array(cal, dtype=np.float32)
        info_arr = np.array(info, dtype=np.float32)
        # Save compressed
        np.savez_compressed(filename, tkr=tkr_arr, cal=cal_arr, meta=info_arr)
        print(f'Saved {filename} with {len(info_arr)} events.')