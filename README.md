# Weather Prediction MLOps

This project is part of the "Cloud Computing" course (RSO) and focuses on Machine Learning in the cloud for weather prediction.

## Project Structure

The project includes the following components:

- `dags`: Apache Airflow DAGs for ETL (Extract, Transform, Load), evaluation, and training.
  - `etl.py`: ETL DAG for data processing.
  - `eval.py`: Evaluation DAG for model evaluation.
  - `train.py`: Training DAG for model training.

- `data`: Data files obtained from the Open-Meteo weather API.

- `models`: Trained machine learning models in `.h5` format. The `.best` extension indicates the best-performing model.

- `open-meteo-api.ipynb`: A Jupyter Notebook containing code for fetching, processing, and loading data into BigQuery from the Open-Meteo API.

- `weather-pred-nn.ipynb`: A Jupyter Notebook that includes exploratory data analysis and training different types of machine learning models (naive, autoregressor, neural network).

## Setting Up Apache Airflow

To set up Apache Airflow for this project, follow these steps:

1. Create a virtual environment and activate it:

   ```bash
   python3 -m venv airflow-venv
   source airflow-venv/bin/activate
   ```

2. Install the required Python packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the Airflow database:

   ```bash
   airflow db init
   ```

4. Create an admin user for Airflow:

   ```bash
   airflow users create \
       --username admin \
       --firstname FIRST_NAME \
       --lastname LAST_NAME \
       --role Admin \
       --email admin@example.com
   ```

5. Set the Google Cloud service account key file path as an environment variable:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

6. Start the Airflow webserver on port 8080:

   ```bash
   airflow webserver -p 8080
   ```

7. Start the Airflow scheduler:

   ```bash
   airflow scheduler -S .
   ```

8. If needed, you can forcefully kill the Airflow webserver and its processes on port 8081:

   ```bash
   sudo fuser -k 8081/tcp
   ```

9. **Important:** Before running the DAGs, set the necessary Airflow variables via the Airflow Web UI. Specifically, add the following variable:

   ```json
   gcp_info = {
       "project_id": "balmy-apogee-404909",
       "bucket_name": "europe-central2-rso-ml-airf-05c3abe0-bucket",
       "dataset_id": "weather_prediction",
       "weather_table_id": "weather_history_LJ",
       "predict_table_id": "weather_predictions",
       "model_table_id": "weather_models"
   }
   ```

   This variable is essential for the correct functioning of the DAGs, as it provides necessary configuration information.


Now, you should have Apache Airflow set up and running for your project.
