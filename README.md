# Customer Churn Prediction

## Running the training pipeline

### On local UNIX (Mac/Linux)
Requirement: Python3.8+

#### Setting up Kaggle
- Get kaggle.json key from [kaggle](https://www.kaggle.com/docs/api): Copy kaggle.json to ```/User/{username}/.kaggle``` (mac) or  ```/home/{username}/.kaggle``` (linux)

#### Installing Requirements
```pip install -r requirements.txt```

####  Setting up Airflow
Airflow needs to be set up separately. Below lists the instructions for setting up airflow locally.
```AIRFLOW_VERSION=2.9.3

# Extract the version of Python you have installed. If you're currently using a Python version that is not supported by Airflow, you may want to set this manually.
# See above for supported versions.
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
# For example this would install 2.9.3 with python 3.8: https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.8.txt

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

Find the airflow.cfg file, it is located in ```/Users/{username}/airflow``` directory in Mac or ```/home/``` in case of Linux.
Change the ```dags_folder```

```
vi {AIRFLOW_PATH}/airflow.cfg
## change this line
dags_folder = {PATH_TO_PROJECT}/dags
```

```commandline
airflow db init
airflow scheduler
airflow webserver
```

Go to [0.0.0.0/8080](0.0.0.0/8080) 
Use the UI to run training and predicion pipeline

### Using Docker
```
docker-compose up airflow-init
docker-compose up
```
Login using "admin" as username and "airflow" as password.