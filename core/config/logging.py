
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Deprecated call to", category=DeprecationWarning)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def setup_logging(level: int | str = logging.INFO, log_file: str = None, console_output: bool = True):
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-7s] %(name)-25s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_dir = Path("log")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / log_path.name
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(file_handler)

    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(level=level, handlers=handlers, force=True)

    for noisy_logger in [
        "chromadb", "httpx", "urllib3", "openai",
        "backoff", "posthog",
        "sentence_transformers",       # 
        "huggingface_hub",             # HF Hub 
        "transformers",                # 
        "torch",                       # PyTorch 
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.getLogger("posthog").setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
