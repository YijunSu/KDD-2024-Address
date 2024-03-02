import os
from loguru import logger


def mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except Exception as _:
        pass


def mknod(file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        logger.debug(f"make file {file_path}")

