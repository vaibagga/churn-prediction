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
docker build -t airflow_fastapi_tensorboard .
docker run -p 8000:8000 -p 8080:8080 -p 6006:6006 -v $(pwd)/logs:/logs -v $(pwd)/dags:/app/airflow/dags airflow_fastapi_tensorboard
```
Login using "admin" as username and "admin" as password.

# Running the inference server
## On local UNIX (Mac/Linux)
Note: If running on local, change the ```BASE_PATH``` in ```config.py``` to the project directory.
The inference server can be run only when ```predict.py``` step has been run atleast once.
To run the FastAPI page, ensure the Redis server is running and start the FastAPI server.
```commandline
uvicorn main:app --reload
```
Go to http://127.0.0.1:8000/docs

## Using Docker
Follow the same instructions as training pipeline, as it installs and starts all the required services. Go to the same URL mentioned above.

# Running tensorboard
## On local UNIX (Max/Linux)
The logs for tensorboard are stored in ```tensorboard/``` directory. To run the board, run
```commandline
 tensorboard --logdir=tensorboard
```
Go to http://localhost:6006
## Using Docker
Run the docker service and go to the same URL.