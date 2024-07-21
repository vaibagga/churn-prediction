import logging

import pandas as pd
import redis
from config import BASE_PATH, MODEL_PATH, KAGGLE_PATH, FILE_NAME, delete_columns
from utils import read_latest_model, read_dataframe_excel, get_latest_path_by_date


def predict_on_batch(df, model):
    df['total_charges'] = df.apply(
        lambda x: x["total_charges"] if x["total_charges"] != ' ' else x["monthly_charges"] * x["tenure_months"],
        axis=1)
    df_ = df.drop(delete_columns, axis=1)
    prediction = model.predict(df_)
    df["prediction"] = prediction
    return df


def save_predictions(df):
    r = redis.Redis(host='localhost', port=6379, db=0)
    key_val = zip(df["customerid"], df["prediction"])
    key_value_dict = dict(key_val)
    r.mset(key_value_dict)


def main():
    df_path, date = get_latest_path_by_date(BASE_PATH, KAGGLE_PATH)
    logging.info(f"Found training data at {df_path}/{date}")
    df = read_dataframe_excel(df_path, date, FILE_NAME)
    model = read_latest_model(BASE_PATH, MODEL_PATH)
    prediction = predict_on_batch(df, model)
    save_predictions(prediction)


if __name__ == "__main__":
    main()
