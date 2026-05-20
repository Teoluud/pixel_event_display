class Event:
    """ Parent class for the whole event. 
    """

    def __init__(self, run_id: int, event_id: int, energy: float, config: dict) -> None:
        """ Constructor.
        """
        # Set event variables
        self.run_id         = run_id
        self.event_id       = event_id
        self.total_energy   = energy
        self.mc_energy = None
        # Load geometry from config
        self.config = config
        self.BIN_HEIGHT = self.config['geometry']['event']['bin_height_mm']
        self.BIN_WIDTH  = self.config['geometry']['event']['bin_width_mm']

    def set_mc_energy(self, mc_energy: float) -> None:
        """ Sets the MC Energy (if available).
        """
        self.mc_energy = mc_energy
