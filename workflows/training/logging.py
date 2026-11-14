"""Console and persistent text logs for the editable training scripts."""

import logging
import sys
from pathlib import Path


def training_logger(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(directory.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(directory / "training.log"),
    ):
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger
