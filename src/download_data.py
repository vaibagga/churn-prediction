import logging
import os.path
import sys
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
try:
    import kaggle
except IOError:
    logging.error("KAGGLE AUTHENTICATION FAILED")
    logging.error("Move kaggle.json to directory ~/HOME/.kaggle")
    sys.exit(1)

from src.config import *


def main():
    DATE = datetime.today().strftime('%Y-%m-%d')
    download_path = f"../{KAGGLE_PATH}/{DATE}/"
    logging.info(f"Donwloading data from Kaggle for {DATE}")
    if os.path.exists(download_path):
        logging.info(f"Data already exists for {DATE}")
        sys.exit(0)
    kaggle.api.dataset_download_files(DATASET_NAME, path=download_path, unzip=True)
    logging.info(f"Data Written at download_path")


if __name__ == "__main__":
    main()
