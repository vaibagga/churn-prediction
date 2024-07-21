## data config
import os

DATASET_NAME = "yeanzc/telco-customer-churn-ibm-dataset"
KAGGLE_FILE = "kaggle.json"
KAGGLE_PATH = "/data/kaggle_data"
BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PREPROCESS_PATH = "/data/preprocessing"
MODEL_PATH = "/model/"
DATA_PATH = "data"
REPORT_PATH = "/reports/"
FILE_NAME = "Telco_customer_churn.xlsx"
TENSORBOARD_PATH = "tensorboard/"

cat_features = ['senior_citizen', 'partner', 'dependents', 'internet_service',
                'online_security', 'online_backup', 'device_protection',
                'tech_support', 'contract', 'paperless_billing', 'payment_method']
num_features = ['tenure_months', 'monthly_charges', 'churn_score', "total_charges"]
target_feature = ['churn_value']
delete_columns = ["customerid", "lat_long", "count", "country", "state", "city", "zip_code", "latitude", "longitude",
                  "city", "churn_label", "gender", "phone_service", "multiple_lines", "churn_reason",
                  "streaming_movies", "streaming_tv", "cltv"]

MIN_TRAINING_DATA_SIZE = 5000