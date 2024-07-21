import datetime
import logging
import sys
import time

from sklearn.model_selection import train_test_split

from utils import *
from config import *
from tensorboardX import SummaryWriter

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    ## tensorboard log directory
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir=f"{BASE_PATH}/{TENSORBOARD_PATH}/{log_dir}")

    ## reading latest file
    df_path, date = get_latest_path_by_date(BASE_PATH, KAGGLE_PATH)
    logging.info(f"Found training data at {df_path}/{date}")
    df = read_dataframe_excel(df_path, date, FILE_NAME)

    ## raise error if data is below limit
    check_df_size(df, MIN_TRAINING_DATA_SIZE)
    logging.info(f"Dataframe size = {df.shape[0]}")
    writer.add_scalar('preprocessing/kaggle_size', df.shape[0], int(time.time()))
    logging.info(f"Dataframe features = {df.shape[1]}")
    writer.add_scalar('preprocessing/kaggle_num_features', df.shape[1], int(time.time()))

    ## drop columns not used for training


    ## log NAs
    num_nas = df.isna().sum(axis=0)
    logging.info(f"Number of NAs {num_nas.sum()}")
    writer.add_scalar('preprocessing/num_nas', num_nas.sum(), int(time.time()))

    ## dealing with empty strings
    df['total_charges'] = df.apply(
        lambda x: x["total_charges"] if x["total_charges"] != ' ' else x["monthly_charges"] * x["tenure_months"],
        axis=1)

    ## train-val-test split of 60-20-20 percent
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_feature])

    ## writing artifacts
    save_file_with_date(train, BASE_PATH, PREPROCESS_PATH, "train.csv", date)
    # save_file_with_date(val, BASE_PATH, PREPROCESS_PATH, "val.csv", date)
    save_file_with_date(test, BASE_PATH, PREPROCESS_PATH, "test.csv", date)


if __name__ == "__main__":
    main()
