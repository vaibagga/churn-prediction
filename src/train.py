import logging
import datetime
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from tensorboardX import SummaryWriter

from config import KAGGLE_PATH, BASE_PATH, PREPROCESS_PATH, TENSORBOARD_PATH, target_feature, num_features, \
    cat_features, MODEL_PATH, REPORT_PATH, delete_columns
from utils import get_latest_path_by_date, read_dataframe_csv, ConvertNonNumericToNaN, ClfSwitcher, save_model, \
    save_file_with_date
from model_grid import hyper_parameters

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    ## tensorboard summary
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir=f"{BASE_PATH}/{TENSORBOARD_PATH}/{log_dir}")

    ## read preprocessed data
    df_path, date = get_latest_path_by_date(BASE_PATH, PREPROCESS_PATH)
    logging.info(f"Found preprocessing data at {df_path}/{date}")
    train = read_dataframe_csv(df_path, date, "train.csv")
    test = read_dataframe_csv(df_path, date, "test.csv")
    ## log metrics to tfboard
    writer.add_scalar('training/train_size', train.shape[0], int(time.time()))
    # writer.add_scalar('training/val_size', val.shape[0])
    writer.add_scalar('training/test_size', test.shape[0], int(time.time()))

    categorical_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_preprocessor = Pipeline(steps=[
        ('convert_to_nan', ConvertNonNumericToNaN(num_features)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_preprocessor, num_features),
            ('cat', categorical_preprocessor, cat_features)
        ])

    # Create pipeline
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('clf', ClfSwitcher())])
    train_sans_col = train.drop(delete_columns, axis=1)
    test_sans_col = test.drop(delete_columns, axis=1)

    X_train, y_train = train_sans_col.drop(target_feature, axis=1), train_sans_col[target_feature]
    X_test, y_test = test_sans_col.drop(target_feature, axis=1), test_sans_col[target_feature]

    gscv = GridSearchCV(pipeline, hyper_parameters, cv=5, n_jobs=-1, return_train_score=True, verbose=3)
    gscv.fit(X_train, y_train)
    best_model = gscv.best_estimator_
    y_pred = best_model.predict(X_test)
    test["prediction"] = y_pred
    misclassified = test[test["prediction"] != test["churn_value"]]
    test_score = best_model.score(X_test, y_test)
    f1 = f1_score(y_pred, y_test)
    writer.add_scalar('training/test_acc', test_score, int(time.time()))
    writer.add_scalar('training/f1', f1, int(time.time()))
    report = classification_report(y_test, y_pred, output_dict=True)
    best_model_performance = pd.DataFrame(report).transpose()

    results = pd.DataFrame(gscv.cv_results_)
    save_file_with_date(results, BASE_PATH, REPORT_PATH, "report.csv", date)
    save_file_with_date(misclassified, BASE_PATH, REPORT_PATH, "misclassified.csv", date)
    save_model(best_model, BASE_PATH, MODEL_PATH, date, "classifier.pkl")
    save_file_with_date(best_model_performance, BASE_PATH, REPORT_PATH, "best_model_report.csv", date)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig(f'{BASE_PATH}/{REPORT_PATH}/{date}/confusion_matrix.png')


if __name__ == "__main__":
    main()
