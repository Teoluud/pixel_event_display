import sys
import logging
import numpy as np
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Chunk:
    """ A structure holding chunk data.
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


class ChunkMerger:
    """ A helper class to merge multiple chunk files into one.
    """

    def __init__(self, chunk_filename_list: list[str]) -> None:
        """ Constructor.
        """
        self.chunk_list = [Chunk(filename) for filename in chunk_filename_list]

    def merge(self) -> None:
        """ Merge the chunks together.
        """
        if not self.chunk_list:
            logger.error("No chunks provided to merge.")
            return
        x_list = [chunk.view_x for chunk in self.chunk_list]
        y_list = [chunk.view_y for chunk in self.chunk_list]
        top_list = [chunk.view_top for chunk in self.chunk_list]
        meta_list = [chunk.meta for chunk in self.chunk_list]
        # Create the numpy arrays
        self.merged_view_x = np.concatenate(x_list, axis=0)
        self.merged_view_y = np.concatenate(y_list, axis=0)
        self.merged_view_top = np.concatenate(top_list, axis=0)
        self.merged_meta = np.concatenate(meta_list, axis=0)

    def save(self, filename: str) -> None:
        """ Save to npz file.
        """
        np.savez_compressed(
            filename,
            view_x=np.array(self.merged_view_x, dtype=np.float32),
            view_y=np.array(self.merged_view_y, dtype=np.float32),
            view_top=np.array(self.merged_view_top, dtype=np.float32),
            meta=np.array(self.merged_meta, dtype=np.float32)
        )
        logger.info(f'Saved {filename} with {len(self.merged_meta)} events.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge list of compressed LAT chunks.')
    # Using nargs='+' automatically captures multiple inputs into a list of strings
    parser.add_argument(
        'files', 
        nargs='+', 
        type=str, 
        help='Paths to the .npz chunk files to merge'
    )
    parser.add_argument(
        '-o', '--outfile', 
        type=str, 
        required=True, 
        help='Path for the saved merged .npz file'
    )

    args = parser.parse_args()

    merger = ChunkMerger(args.files)
    merger.merge()
    merger.save(args.outfile)