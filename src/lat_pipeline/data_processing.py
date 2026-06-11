import numpy as np

from .stream_parser import StreamParser
from .import_data import ImportData
from .event_display import EventDisplay
from .utils import normalize_log_matrix
from .merit_reprocessing import merit_reprocessing


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

    def _stack_side_view(self, tkr_matrix: np.ndarray, cal_matrix: np.ndarray) -> np.ndarray:
        """ Helper to stack a side projection.
        """
        raw_tkr = np.ma.filled(tkr_matrix, fill_value=0.0)
        raw_cal = np.ma.filled(cal_matrix, fill_value=0.0)
        # Create the physical gap between CAL and TKR (3-pixel gap for geometric accuracy (~75mm))
        gap_pad = np.zeros((3, 113), dtype=np.float32)
        # Stack vertically: CAL (bottom), Gap, TKR (top)
        # Total rows = 8 + 3 + 18 = 27 rows
        combined_matrix = np.vstack((raw_cal, gap_pad, raw_tkr))
        rows = np.size(combined_matrix, axis=0)
        columns = np.size(combined_matrix, axis=1)
        # Pad the top to reach exactly 113x113
        top_pad = columns-rows
        square_matrix = np.pad(combined_matrix, pad_width=((0,top_pad), (0,0)), mode='constant', constant_values=0.0)
        return square_matrix

    def process_and_save(self, output_prefix: str = 'dataset') -> None:
        """ Runs the full pipeline for a specific projection in a batch.
        """
        matrices_x = []
        matrices_y = []
        matrices_top = []
        event_infos = []    # Store run_id, event_id, energy as metadata
        merit_values = []
        merit_names = []
        chunk_index = 0
        event_count = 0
        # Iterate through the piped stream dinamically
        for event, tkr, cal, merit_vars in self.stream_parser.parse_stream():
            merit_vars = merit_reprocessing(merit_vars, event.total_energy)
            # Process X-Z view
            tkr_x, _, _ = tkr.get_matrix('x')
            cal_x, _, _ = cal.get_matrix_side('x')
            combined_x = self._stack_side_view(tkr_x, cal_x)
            # Process Y-Z view
            tkr_y, _, _ = tkr.get_matrix('y')
            cal_y, _, _ = cal.get_matrix_side('y')
            combined_y = self._stack_side_view(tkr_y, cal_y)
            # Process X-Y view (CAL only)
            cal_top, _, _ = cal.get_matrix_top()
            raw_top = np.ma.filled(cal_top, fill_value=0.0)
            # Append to chunk lists
            matrices_x.append(combined_x)
            matrices_y.append(combined_y)
            matrices_top.append(raw_top)
            event_infos.append([event.run_id, event.event_id, event.total_energy, event.mc_energy])
            merit_names.append(list(merit_vars.keys()))
            merit_values.append([merit_vars[key] for key in merit_names[event_count]])
            event_count += 1
            # Save and flush memory when chunk is full
            if event_count >= self.chunk_size:
                self._save_chunk(matrices_x, matrices_y, matrices_top, event_infos, merit_values, merit_names, output_prefix, chunk_index)
                # Reset lists to free up RAM
                matrices_x, matrices_y, matrices_top, event_infos = [], [], [], []
                chunk_index += 1
                event_count = 0
        # Save any remaining events after the pipe closes
        if event_count > 0:
            self._save_chunk(matrices_x, matrices_y, matrices_top, event_infos, merit_values, merit_names, output_prefix, chunk_index)

    def _save_chunk(self, mat_x, mat_y, mat_top, info, vars, vars_names, prefix, index):
        """ Helper to save a list of matrices to a compressed numpy archive.
        """
        filename = f'{prefix}_chunk_{index:04d}.npz'
        # Save compressed
        np.savez_compressed(
            filename,
            view_x=np.array(mat_x, dtype=np.float32),
            view_y=np.array(mat_y, dtype=np.float32),
            view_top=np.array(mat_top, dtype=np.float32),
            meta=np.array(info, dtype=np.float32),
            merit_values=np.array(vars, dtype=np.float32),
            merit_names=np.array(vars_names, dtype=str)
        )
        print(f'Saved {filename} with {len(info)} events.')