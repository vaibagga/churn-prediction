import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import re

import sklearn
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt
import seaborn as sns


def read_dataframe_excel(base_path: str, data_path: str, file_path: str) -> pd.DataFrame:
    full_path = f"{base_path}/{data_path}/{file_path}"
    df = pd.read_excel(full_path)
    df.columns = df.columns.map(lambda x: re.sub('\s+', '_', x.lower().strip()))
    for col in df.columns.values:
        df[col].replace('', np.nan, inplace=True)
    return df


def read_dataframe_csv(base_path: str, data_path: str, file_path: str) -> pd.DataFrame:
    full_path = f"{base_path}/{data_path}/{file_path}"
    df = pd.read_csv(full_path)
    return df


def get_latest_path_by_date(base_path: str, dir_path: str, return_none=False):
    path = base_path + dir_path
    file_list = os.listdir(path)
    if len(file_list) == 0:  # no subdirectory present
        if return_none:
            return path, None
        logging.exception(f"Empty directory {path}")
        sys.exit(1)
    dir_list = sorted(file_list)
    return path, dir_list[-1]


def check_df_size(df: pd.DataFrame, min_size: int) -> None:
    if len(df) < min_size:
        logging.error(f"Dataframe has only {df.shape[0]} required")
        logging.error(f"Minimum {min_size} rows required")
        sys.exit(1)


def save_file_with_date(dataframe: pd.DataFrame, base_path: str, save_path: str, file_name: str, date):
    full_path = f"{base_path}{save_path}/{date}"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    file_path = f"{full_path}/{file_name}"
    if os.path.isfile(full_path):
        logging.exception(f"File already exists for {file_path}")
        sys.exit(1)
    logging.info(f"Writing dataframe with {len(dataframe)} rows to {file_path}")
    dataframe.to_csv(file_path, index=False)


class ConvertNonNumericToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.numeric_features:
            X[feature] = pd.to_numeric(X[feature], errors='coerce')
        return X


# class DataTransformer(TransformerMixin):
#     def __init__(self, cat_features_=None, numeric_features_=None):
#         self.label_encoders = {}
#         self.scalers = {}
#         self.imputers = {}
#         self.cat_features = cat_features_
#         self.numeric_features = numeric_features_
#
#     def fit(self, df: pd.DataFrame) -> None:
#         for cf in self.cat_features:
#             le = LabelEncoder()
#             imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#             df[cf] = le.fit_transform(df[cf])
#             df[cf] = le.fit_transform(df[cf])
#             self.label_encoders[cf] = le
#             self.imputers[cf] = imputer
#         scaler = preprocessing.StandardScaler()
#         df[self.numeric_features] = scaler.fit_transform(df[self.numeric_features])
#         imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#         df[self.numeric_features] = scaler.fit_transform(df[self.numeric_features])
#         df[self.numeric_features] = imputer.fit_transform(df[self.numeric_features])
#         self.scalers = scaler
#         self.imputers[self.numeric_features] = imputer
#
#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         for cf in self.cat_features:
#             df[cf] = self.label_encoders[cf].transform(df[cf])
#             df[cf] = self.imputers[cf].transform(df[cf])
#         for nf in self.numeric_features:
#             df[nf] = self.scalers[nf].transform(df[nf])
#             df[nf] = self.imputers[nf].transform(df[nf])
#         return df

def save_model(model_pipeline: sklearn.pipeline, base_path: str, model_path: str, date: str, file_name: str) -> None:
    full_dir = f"{base_path}/{model_path}/{date}"
    full_path = f"{base_path}/{model_path}/{date}/{file_name}"
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    joblib.dump(model_pipeline, full_path)


def read_latest_model(base_path: str, model_path):
    model_path, date = get_latest_path_by_date(base_path, model_path)
    full_path = f"{model_path}/{date}/classifier.pkl"
    model = joblib.load(full_path)
    return model


class ClfSwitcher(BaseEstimator):
    def __init__(self, estimator=SGDClassifier()):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
