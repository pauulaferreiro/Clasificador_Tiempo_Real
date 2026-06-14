#logger_config.py
import logging
import time
from functools import wraps
from pathlib import Path


def setup_logger(log_name: str = "video_classifier.log"):

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("VideoAI")
    logger.setLevel(logging.INFO)

    # Evitar duplicados 
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_dir / log_name,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


log = setup_logger()

def setup_latency_logger(log_name: str = "latency.log"):
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger("LatencyLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        file_handler = logging.FileHandler(log_path / log_name, encoding='utf-8')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


latency_log = setup_latency_logger()

def monitor_latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            latency = time.perf_counter() - start_time

            log.info(f"LATENCIA | {func.__name__} ejecutado en {latency:.4f}s")
            latency_log.info(f"{func.__name__},{latency:.6f}")

            return result

        except Exception as e:
            latency = time.perf_counter() - start_time

            log.error(
                f"ERROR | {func.__name__} tras {latency:.4f}s - {str(e)}"
            )

            raise e

    return wrapper


def monitor_inference(name: str = "inference"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            result = func(*args, **kwargs)

            elapsed = time.perf_counter() - start

            log.info(f"{name.upper()} | {elapsed:.4f}s")

            return result

        return wrapper
    return decorator