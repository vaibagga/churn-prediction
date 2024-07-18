import datetime
import logging
import sys

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
    writer.add_scalar('preprocessing/kaggle_size', df.shape[0])
    logging.info(f"Dataframe features = {df.shape[1]}")
    writer.add_scalar('preprocessing/kaggle_num_features', df.shape[1])

    ## drop columns not used for training
    df.drop(delete_columns, axis=1, inplace=True)

    ## log NAs
    num_nas = df.isna().sum(axis=0)
    logging.info(f"Number of NAs {num_nas.sum()}")
    writer.add_scalar('preprocessing/num_nas', num_nas.sum())


    ## train-val-test split of 60-20-20 percent
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_feature])
    train, val = train_test_split(train, test_size=0.25, random_state=42,
                                                      stratify=train[target_feature])

    ## writing artifacts
    save_file_with_date(train, BASE_PATH, PREPROCESS_PATH, "train.csv", date)
    save_file_with_date(val, BASE_PATH, PREPROCESS_PATH, "val.csv", date)
    save_file_with_date(test, BASE_PATH, PREPROCESS_PATH, "test.csv", date)




if __name__ == "__main__":
    main()
