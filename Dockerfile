# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/app/airflow

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required packages
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir tensorboard supervisor apache-airflow

COPY ./kaggle.json /.kaggle/

# Initialize Airflow database
RUN airflow db init
RUN airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# Make ports 8000, 8080, and 6006 available to the world outside this container
EXPOSE 8000 8080 6006

# Command to run supervisord
CMD ["supervisord", "-c", "/app/supervisord.conf"]
