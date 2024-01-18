import datetime
from airflow.decorators import dag, task
from airflow.models import Variable
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

default_args = {
    'owner': 'user',
    'start_date': datetime.datetime(2023, 1, 1),
    'email': ['drejcpesjak.pesjak@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

@dag(
    dag_id='eval_dag',
    default_args=default_args, 
    schedule_interval=None, 
    catchup=False
)
def eval_dag():
    @task
    def calc_rmse(model_id_name, gcp_info):
        print(model_id_name)

        PROJECT_ID = gcp_info['project_id']
        DATASET_ID = gcp_info['dataset_id']
        WEATHER_TABLE_ID = gcp_info['weather_table_id']
        PREDICT_TABLE_ID = gcp_info['predict_table_id']
        MODEL_TABLE_ID = gcp_info['model_table_id']

        client = bigquery.Client(project=PROJECT_ID)

        # Query to fetch the last month of weather data
        query_weather = f"""
            SELECT * 
            FROM `{PROJECT_ID}.{DATASET_ID}.{WEATHER_TABLE_ID}`
            WHERE DATE(time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
        """
        weather_data = client.query(query_weather).to_dataframe()

        # Query to fetch the last month of predictions
        query_predict = f"""
            SELECT * 
            FROM `{PROJECT_ID}.{DATASET_ID}.{PREDICT_TABLE_ID}`
            WHERE DATE(weather_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
            AND model_id_name = '{model_id_name}'
        """
        predict_data = client.query(query_predict).to_dataframe()

        # Merge datasets
        merged_data = pd.merge(weather_data, predict_data, left_on='time', right_on='weather_date')

        # Check if merged data is available
        if merged_data.empty:
            raise ValueError("No overlapping data between weather and predictions.")

        y_true = merged_data[['temperature_2m_mean____C_', 'precipitation_sum__mm_']]
        y_pred = merged_data[['temp_predict', 'precip_predict']]

        # Calculate RMSE for each column
        rmse_temp = np.sqrt(mean_squared_error(y_true['temperature_2m_mean____C_'], y_pred['temp_predict']))
        rmse_precip = np.sqrt(mean_squared_error(y_true['precipitation_sum__mm_'], y_pred['precip_predict']))
        rmse = (rmse_temp + rmse_precip) / 2

        train_datetime = model_id_name.split('_')[-1]
        train_datetime = datetime.datetime.strptime(train_datetime, "%Y%m%d-%H%M%S")

        # Prepare data for BigQuery upload
        data = [{
            'model_id_name': model_id_name,
            'train_datetime': train_datetime,
            'rmse_temp': rmse_temp,
            'rmse_precip': rmse_precip,
            'rmse': rmse
        }]
        dataframe = pd.DataFrame(data)

        # Upload to BigQuery
        dataset_ref = client.dataset(DATASET_ID)
        table_ref = dataset_ref.table(MODEL_TABLE_ID)
        job = client.load_table_from_dataframe(dataframe, table_ref)
        job.result()  # Wait for the job to complete

        return rmse

    @task
    def get_best(current_model_id, rmse_new, gcp_info):
        PROJECT_ID = gcp_info['project_id']
        DATASET_ID = gcp_info['dataset_id']
        MODEL_TABLE_ID = gcp_info['model_table_id']

        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT model_id_name
            FROM `{PROJECT_ID}.{DATASET_ID}.{MODEL_TABLE_ID}`
            ORDER BY rmse ASC
            LIMIT 1
        """
        result = client.query(query).to_dataframe()
        best_model_id_name = result['model_id_name'].iloc[0]
        print(f'Best model ID: {best_model_id_name}; current model ID: {current_model_id}')
        return best_model_id_name

    @task
    def update_proxy(best_model_id_name, gcp_info):
        BUCKET_NAME = gcp_info['bucket_name']
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Define the new proxy file name
        new_proxy_file_name = f"models/{best_model_id_name}.best"

        # List all blobs in the models/ folder
        blobs = bucket.list_blobs(prefix="models/")
        for blob in blobs:
            if blob.name.endswith(".best"):
                # Rename the existing .best file
                bucket.rename_blob(blob, new_proxy_file_name)
                break

    # Get GCP info from Airflow variables
    # gcp_info = {
    #     "project_id": "balmy-apogee-404909",
    #     "bucket_name": "europe-central2-rso-ml-airf-05c3abe0-bucket",
    #     "dataset_id": "weather_prediction",
    #     "weather_table_id": "weather_history_LJ",
    #     "predict_table_id": "weather_predictions",
    #     "model_table_id": "weather_models"
    # }
    gcp_info = Variable.get('gcp_info', deserialize_json=True)
    
    model_id_name = '{{ dag_run.conf["model_id_name"] }}'
    # model_id_name = "dnn_multitarget_20231205-184325"

    # Define the DAG flow
    rmse = calc_rmse(model_id_name, gcp_info)
    best_model_id_name = get_best(model_id_name, rmse, gcp_info)
    update_proxy(best_model_id_name, gcp_info)

eval_dag_instance = eval_dag()
