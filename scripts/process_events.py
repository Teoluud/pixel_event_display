import sys
import argparse
import logging
import yaml
from typing import Any
from pathlib import Path
# Define working directory
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
sys.path.append(str(SRC_DIR))

from lat_pipeline.data_processing import BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixel Event Display orchestrator.')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to YAML config')

    args = parser.parse_args()
    # Load configuration
    absolute_config_path = ROOT_DIR / args.config
    config = load_config(absolute_config_path)
    logger.info('Configuration loaded successfully.')

    logger.info("Starting Batch Processing Mode from stdin...")
    batch_processor = BatchProcessor(config=config)
    batch_processor.process_and_save()
    logger.info('Batch processing complete.')