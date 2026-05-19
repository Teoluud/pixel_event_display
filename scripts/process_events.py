import sys
import argparse
import logging
import yaml
from pathlib import Path
# Define working directory
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
sys.path.append(str(SRC_DIR))

from lat_pipeline.data_processing import DataProcessing, BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixel Event Display orchestrator.')
    parser.add_argument('--run', type=int, help='Run ID (required for single event)')
    parser.add_argument('--event', type=int, help='Event ID (required for single event)')
    parser.add_argument('--path', type=str, default='../events/mcElectrons/csv', help='Path to import data')
    parser.add_argument('--view', type=str, default='x', help='Projection view')
    parser.add_argument('--batch', action='store_true', help='Process piped input and save to npz')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to YAML config')

    args = parser.parse_args()
    # Load configuration
    absolute_config_path = ROOT_DIR / args.config
    config = load_config(absolute_config_path)
    logger.info('Configuration loaded successfully.')

    if args.batch:
        logger.info("Starting Batch Processing Mode from stdin...")
        batch_processor = BatchProcessor(config=config)
        batch_processor.process_and_save()
        logger.info('Batch processing complete.')
    else:
        if args.run is None or args.event is None:
            logger.error('Single event display requires both --run and --event arguments.')
            sys.exit(1)
        logger.info(f'Starting Single Event Display for Run {args.run}, Event {args.event}')
        pipeline = DataProcessing(args.path, args.run, args.event, config=config)
        pipeline.process_and_display(args.view)