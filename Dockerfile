# Use the official Apache Airflow image as the base image
FROM apache/airflow:2.5.0-python3.8

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow

# Install PostgreSQL development packages
USER root
RUN apt-get update && \
    apt-get install -y postgresql-client libpq-dev && \
    apt-get clean

# Switch back to airflow user
USER airflow

# Copy the project code files into the Docker image
COPY . $AIRFLOW_HOME

# Install any additional dependencies
RUN pip install poetry
RUN pip install --no-cache-dir --no-build-isolation -r $AIRFLOW_HOME/requirements.txt

# Initialize the Airflow database
RUN airflow db init

# Expose the port for the web server
EXPOSE 8080

# Create the default user
RUN airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Set the entrypoint to start the Airflow web server and scheduler
CMD ["sh", "-c", "airflow scheduler & airflow webserver"]
