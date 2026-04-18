class Event:
    """ Parent class for the whole event. 
    """
    BIN_HEIGHT: float   = 17    # mm
    BIN_WIDTH: float    = 17    # mm

    def __init__(self, run_id: int, event_id: int, energy: float) -> None:
        """ Constructor.
        """
        # Set event variables
        self.run_id         = run_id
        self.event_id       = event_id
        self.total_energy   = energy
