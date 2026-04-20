from import_data import ImportData
from event_display import EventDisplay


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
        # Instantiate the display view
        self.display = EventDisplay(self.run_id, self.event_id)

    def process_and_display(self, view: str = 'x') -> None:
        """
        Runs the full pipeline for a specific projection.
        """
        # Import data
        importer = ImportData(self.path, self.run_id, self.event_id)
        event, tkr, cal = importer.import_event()
        # Extract matrices
        tkr_matrix, tkr_x, tkr_z        = tkr.get_matrix(view)
        cal_matrix_side, cal_x, cal_z   = cal.get_matrix_side(view)
        # Render via the View
        self.display.plot_combined_view(tkr_x, tkr_z, tkr_matrix, cal_x, cal_z, cal_matrix_side, view_name=view)
        # self.display.plot_single_view(tkr_x, tkr_z, tkr_matrix, view_name=view, detector='TKR')
        # self.display.plot_single_view(cal_x, cal_z, cal_matrix_side, view_name=view, detector='CAL')
        self.display.show_all()