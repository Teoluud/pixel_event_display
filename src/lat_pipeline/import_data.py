import pandas as pd

from . import Event, TkrData, CalData


class ImportData:
    """ Class that handles the data importing and the parsing.
    """
    
    def __init__(self, path: str, run_id: str, event_id: str, config: dict) -> None:
        """ Constructor.
        """
        self.run_id     = int(run_id)
        self.event_id   = int(event_id)
        self.config     = config
        # Define file paths
        self.filename_prefix: str    = 'event_data'
        tkr_filename: str       = f'{self.filename_prefix}_TKR_{run_id}_{event_id}.csv'
        self.tkr_path: str      = f'{path}/{tkr_filename}'
        cal_filename: str       = f'{self.filename_prefix}_CAL_{run_id}_{event_id}.csv'
        self.cal_path: str      = f'{path}/{cal_filename}'

    def read_csv(self, filepath: str) -> pd.DataFrame:
        """ Imports the csv file and converts it to pandas dataframe.
        """
        data_frame: pd.DataFrame = pd.read_csv(filepath, comment='#')
        return data_frame

    def import_event(self) -> tuple[Event, TkrData, CalData]:
        """ Creates the objects needed for one event.
        """
        with open(self.cal_path, "r") as infile:
            line = infile.readline().rstrip()
            energy = float(line.split(": ")[1])
        tkr_dataframe = self.read_csv(self.tkr_path)
        cal_dataframe = self.read_csv(self.cal_path)
        event = Event(self.run_id, self.event_id, energy, config=self.config)
        tkr = TkrData(tkr_dataframe, event, config=self.config)
        cal = CalData(cal_dataframe, event, config=self.config)
        return event, tkr, cal