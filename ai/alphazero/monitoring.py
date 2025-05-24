import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any # Added Any for config type hint flexibility

# Ensure tensorboard is installed: pip install tensorboard
# Handle potential import error for SummaryWriter if tensorboard is not installed.
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None # Allow the program to run, but TensorBoard logging will be disabled

# Import configuration constants
# Assuming config.py is in the same directory (alphazero)
try:
    from .config import TENSORBOARD_DIR, TRAINING_LOG_FILE, LOG_LEVEL
except ImportError:
    # Fallback or default values if config import fails, or raise error
    # This is important if monitoring.py might be used standalone or if config is not yet finalized
    print("Warning: Could not import config from .config in monitoring.py. Using default log paths/levels.")
    TENSORBOARD_DIR = "ai/logs/tensorboard_default"
    TRAINING_LOG_FILE = "ai/logs/training_default.log"
    LOG_LEVEL = "INFO"

class ExperimentLogger:
    _instance: Optional['ExperimentLogger'] = None

    def __new__(cls, *args, **kwargs) -> 'ExperimentLogger':
        if cls._instance is None:
            cls._instance = super(ExperimentLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, run_name: Optional[str] = None, log_to_console: bool = True, config: Optional[Any] = None):
        if self._initialized:
            return

        # Use config values if provided, otherwise use imported/defaulted constants
        _tensorboard_dir = config.TENSORBOARD_DIR if config and hasattr(config, 'TENSORBOARD_DIR') else TENSORBOARD_DIR
        _training_log_file = config.TRAINING_LOG_FILE if config and hasattr(config, 'TRAINING_LOG_FILE') else TRAINING_LOG_FILE
        _log_level = config.LOG_LEVEL if config and hasattr(config, 'LOG_LEVEL') else LOG_LEVEL
        
        os.makedirs(_tensorboard_dir, exist_ok=True)
        # Ensure the directory for TRAINING_LOG_FILE exists
        os.makedirs(os.path.dirname(_training_log_file), exist_ok=True)


        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.run_path = os.path.join(_tensorboard_dir, run_name)
        
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.run_path)
            print(f"TensorBoard SummaryWriter initialized at {self.run_path}")
        else:
            self.writer = None
            print("Warning: SummaryWriter not available (tensorboard not installed?). TensorBoard logging disabled.")

        # Setup Python logger
        self.logger = logging.getLogger("AlphaZeroTrainer")
        # Clear existing handlers (if any, e.g., during re-runs in interactive sessions)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.setLevel(getattr(logging, _log_level.upper(), logging.INFO))

        # File handler
        fh = logging.FileHandler(_training_log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # Console handler
        if log_to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

        self.logger.info(f"Experiment Logger initialized. Python logs: {_training_log_file}. TensorBoard logs (if enabled): {self.run_path}")
        self._initialized = True

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        self.logger.debug(f"Step {step} - Scalar {tag}: {value}")

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        for tag, value in tag_scalar_dict.items():
            self.logger.debug(f"Step {step} - Scalars {main_tag}/{tag}: {value}")

    def log_histogram(self, tag: str, values, step: int, bins='auto'):
        if self.writer:
            self.writer.add_histogram(tag, values, step, bins=bins)
        self.logger.debug(f"Step {step} - Histogram logged for {tag}")

    def log_text(self, tag: str, text_string: str, step: int):
        if self.writer:
            self.writer.add_text(tag, text_string, step)
        # Use info for text logs as they are usually summaries or important events
        self.logger.info(f"Step {step} - Text '{tag}': {text_string}")

    def log_info(self, message: str):
        self.logger.info(message)

    def log_warning(self, message: str):
        self.logger.warning(message)
        
    def log_error(self, message: str, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

    def close(self):
        if self.writer:
            self.writer.close()
        self.logger.info("Experiment Logger closed.")
        # Remove handlers to allow for potential re-initialization if needed
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

def get_experiment_logger() -> ExperimentLogger:
    if ExperimentLogger._instance is None or not ExperimentLogger._instance._initialized:
        # Optionally, initialize with default settings if not done, or enforce prior initialization.
        # For robustness, one might call ExperimentLogger() here if a default setup is acceptable.
        # However, the user prompt implies it should be initialized first (e.g. in train_pipeline.py)
        raise RuntimeError("ExperimentLogger has not been initialized. Call ExperimentLogger() first, typically at the start of your main training script.")
    return ExperimentLogger._instance

# Example of how it might be initialized at the start of a script:
# if __name__ == '__main__':
#     # This is just for testing monitoring.py itself
#     try:
#         from .config import CONFIG_OBJECT # Assuming config is structured to provide a CONFIG_OBJECT or similar
#         logger = ExperimentLogger(run_name="test_run_001", config=CONFIG_OBJECT)
#     except ImportError:
#         logger = ExperimentLogger(run_name="test_run_001_no_config") # Run with defaults if config can't be loaded
    
#     logger.log_info("This is an info message from monitoring test.")
#     logger.log_scalar("test_metric/my_scalar", 0.75, 0)
#     logger.log_scalars("test_metric_group", {"metric1": 0.1, "metric2": 0.2}, 1)
#     logger.log_text("hyperparameters", "lr: 0.001, batch_size: 64", 0)
#     logger.log_warning("This is a test warning.")
#     try:
#         x = 1/0
#     except ZeroDivisionError:
#         logger.log_error("A division by zero occurred.", exc_info=True)
#     logger.close()

#     # Test get_experiment_logger
#     retrieved_logger = get_experiment_logger()
#     if retrieved_logger is logger:
#         print("Singleton access successful.")
#     retrieved_logger.log_info("Logged using retrieved logger instance.")
#     retrieved_logger.close() # Note: closing handlers twice might be an issue if not careful.
                              # The close method should ideally make handlers unusable or clear them fully.
