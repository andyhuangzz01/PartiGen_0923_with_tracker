import logging
from loguru import logger
import sys
import os
from pathlib import Path

from omegaconf import OmegaConf


class HydraLoggerBridge(logging.Handler):

    def emit(self, record):
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth,
                   exception=record.exc_info).log(level, record.getMessage())


def set(cfg):
    """Set up logging for the training session."""
    # Create logs directory if it doesn't exist
    log_dir = Path(cfg.experiment_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    hydra_log_path = os.path.join(cfg.experiment_dir, "run.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    logger.info(f"Logging initialized for experiment: {cfg.expname}")
    logger.info(f"Experiment directory: {cfg.experiment_dir}")

    hydra_cfg_path = os.path.join(cfg.experiment_dir, "cfg.yaml")
    OmegaConf.save(cfg, hydra_cfg_path, resolve=True)

    return logger
