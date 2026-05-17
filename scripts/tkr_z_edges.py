import pandas as pd
import numpy as np
import glob

def get_tkr_z_edges(view_name) -> np.ndarray:
    """ Generates the custom z edges for the TKR, in order to have every plane tray on a single pixel (no gaps).
    """
    all_zs = []
    # Load every TKR csv file available
    for file in glob.glob('/home/matteo/FERMI-LAT/event-display/events/mcElectrons/csv/event_data_TKR_*.csv'):
        df = pd.read_csv(file, comment='#')
        # Filter by X or Y view
        view_df: pd.DataFrame = df[df['View'] == view_name.upper()]
        view_zs = view_df['Z'].to_numpy()
        all_zs.extend(view_zs)
    # Find the exact, unique Z heights of the layers (rounded to avoid float errors)
    unique_layers = np.sort(np.unique(np.round(all_zs, 2)))
    # Calculate exact midpoints between layers to serve as bin edges
    midpoints = (unique_layers[1:] + unique_layers[:-1]) / 2.0
    # Pad the top and bottom edges
    edges = np.concatenate([
        [unique_layers[0] - 15.0],
        midpoints,
        [unique_layers[-1] + 15.0]
    ])
    return edges

if __name__ == '__main__':
    print('tkr_z_edges_x:', get_tkr_z_edges('X').tolist())
    print('tkr_z_edges_y:', get_tkr_z_edges('Y').tolist())