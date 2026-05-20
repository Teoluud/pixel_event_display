import sys
import io

import pandas as pd

from .event import Event
from .tkr_data import TkrData
from .cal_data import CalData


class StreamParser:
    """ 
    Reads pipeline standard input (sys.stdin) line by line.
    Yields parsed Event, TkrData, and CalData objects dynamically.
    """

    def __init__(self, config: dict) -> None:
        """ Constructor.
        """
        self.config = config

    def parse_stream(self):
        """ Generator that yields one event at a time from the pipe.
        """
        current_run_id = None
        current_event_id = None
        current_energy = 0.0
        current_mc_energy = 0.0
        tkr_lines = []
        cal_lines = []
        current_block = None
        # Read line by line from the piped upstream program
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("EVENT_START"):
                # Example line: "EVENT_START Run: 50 Event: 39 Energy: 407600"
                parts = line.split()
                current_run_id = parts[2]
                current_event_id = parts[4]
                current_energy = parts[6]
                current_mc_energy = parts[8]
            elif line.startswith("TKR_START"):
                current_block = 'TKR'
                tkr_lines = []  # Reset for new event
            elif line.startswith("CAL_START"):
                current_block = 'CAL'
                cal_lines = []  # Reset for new event
            elif line.startswith("EVENT_END"):
                # We have all the text for this event, we can process it.
                if tkr_lines and cal_lines and current_run_id is not None and current_event_id is not None:
                    # Convert text lists to single strings, then to string-buffers for pandas
                    tkr_csv = io.StringIO('\n'.join(tkr_lines))
                    cal_csv = io.StringIO('\n'.join(cal_lines))
                    tkr_df = pd.read_csv(tkr_csv)
                    cal_df = pd.read_csv(cal_csv)
                    event = Event(int(current_run_id), int(current_event_id), float(current_energy), config=self.config)
                    event.set_mc_energy(float(current_mc_energy))
                    tkr = TkrData(tkr_df, event, config=self.config)
                    cal = CalData(cal_df, event, config=self.config)
                    yield event, tkr, cal
            # Collect data based on the current block
            else:
                if current_block == 'TKR':
                    tkr_lines.append(line)
                elif current_block == 'CAL':
                    cal_lines.append(line)