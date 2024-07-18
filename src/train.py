import logging
import datetime

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorboardX import SummaryWriter

from config import KAGGLE_PATH, BASE_PATH, PREPROCESS_PATH, TENSORBOARD_PATH, target_feature, num_features, cat_features
from utils import get_latest_path_by_date, read_dataframe_csv, ConvertNonNumericToNaN, ClfSwitcher

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
    val = read_dataframe_csv(df_path, date, "val.csv")
    test = read_dataframe_csv(df_path, date, "test.csv")
    ## log metrics to tfboard
    writer.add_scalar('training/train_size', train.shape[0])
    writer.add_scalar('training/val_size', val.shape[0])
    writer.add_scalar('training/test_size', test.shape[0])

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
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier'), ClfSwitcher()])

    X_train, y_train = train.drop(target_feature, axis=1), train[target_feature]
    X_val, y_val = val.drop(target_feature, axis=1), val[target_feature]
    X_test, y_test = test.drop(target_feature, axis=1), test[target_feature]

    # Fit the pipeline
    parameters = [
        {
            'clf__estimator': [LogisticRegression()],  # SVM if hinge loss / logreg if log loss
            'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
            'tfidf__stop_words': ['english', None],
            'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
            'clf__estimator__max_iter': [50, 80],
            'clf__estimator__tol': [1e-4],
            'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
        },
        {
            'clf__estimator': [MultinomialNB()],
            'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
            'tfidf__stop_words': [None],
            'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
        },
    ]

    gscv = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=False, verbose=3)
    gscv.fit(X_train, y_train)

    if __name__ == "__main__":
        main()
