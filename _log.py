"""Shared logging setup. Each script calls: log = setup_log("name")"""

import logging
from pathlib import Path


def setup_log(name: str) -> logging.Logger:
    Path("log").mkdir(exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(f"log/{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Only warnings+ go to terminal (prints handle normal output)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    return log
