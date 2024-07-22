import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 7, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'churn_prediction_inference',
    default_args=default_args,
    description='A DAG to predict churn on a batch',
    schedule_interval=timedelta(days=1),
)
base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Define the paths to the Python scripts
predict_script_path = f'/app/src/predict.py'


def run_python_command(file_path, task_id):
    cmd_command = f"python3 {file_path}"
    return BashOperator(
        task_id=task_id,
        bash_command=cmd_command,
        dag=dag)


prediction_task = run_python_command(predict_script_path, "predict_task")

