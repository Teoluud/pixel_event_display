from import_data import ImportData
from event_display import EventDisplay
from utils import normalize_log_matrix


class DataProcessing:
    """ 
    High-level orchestrator class. 
    It commands the ImportData (Controller), fetches processed matrices, and pushes them to the EventDisplay (View).
    """
    def __init__(self, path: str, run_id: str, event_id: str) -> None:
        """
        Constructor.
        """
        self.path       = path
        self.run_id     = run_id
        self.event_id   = event_id

    def process_and_display(self, view: str = 'x') -> None:
        """
        Runs the full pipeline for a specific projection.
        """
        # Import data
        importer = ImportData(self.path, self.run_id, self.event_id)
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