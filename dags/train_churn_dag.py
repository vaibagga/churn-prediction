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
    'churn_prediction',
    default_args=default_args,
    description='A DAG to run multiple Python scripts',
    schedule_interval=timedelta(days=1),
)

# Define the paths to the Python scripts
download_script_path = '/opt/airflow/src/download_data.py'
split_script_path = '/opt/airflow/src/split_data.py'
train_script_path = '/opt/airflow/src/split_data.py'


def run_python_command(file_path, task_id):
    cmd_command = f"python3 {file_path}"
    return BashOperator(
        task_id=task_id,
        bash_command=cmd_command,
        dag=dag)


download_task = run_python_command(download_script_path, "download_task")
split_task = run_python_command(split_script_path, "split_task")
train_task = run_python_command(train_script_path, "train_task")

download_task >> split_task >> train_task
