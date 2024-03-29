import datetime
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
# from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
# from airflow.providers.google.cloud.sensors.gcs import GCSObjectsWithPrefixExistenceSensor
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from google.cloud import bigquery

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
    dag_id='train_dag',
    default_args=default_args, 
    schedule_interval=datetime.timedelta(days=1), 
    catchup=False
)
def weather_prediction_training():

    @task
    def set_model_id_name():
        name = 'dnn_multitarget'
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_id_name = f"{name}_{timestamp}"
        # MODEL_PATH = f"gs://{BUCKET_NAME}/models/{model_id_name}.h5"
        return model_id_name

    @task
    def fetch_weather_data(gcp_info):
        PROJECT_ID = gcp_info['project_id']
        DATASET_ID = gcp_info['dataset_id']
        WEATHER_TABLE_ID = gcp_info['weather_table_id']

        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT * 
            FROM `{PROJECT_ID}.{DATASET_ID}.{WEATHER_TABLE_ID}`
            ORDER BY time
        """
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df

    @task
    def split_data(df: pd.DataFrame):
        # Assuming preprocessing in ETL pipeline

        # set time as index
        df = df.set_index('time')

        # Split the data, last 10 rows as test set
        test_size = 10
        X_train, X_test = df.iloc[:-test_size], df.iloc[-test_size:]
        y_train = X_train[['temperature_2m_mean____C_', 'precipitation_sum__mm_']]
        y_test = X_test[['temperature_2m_mean____C_', 'precipitation_sum__mm_']]

        # convert to float32
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
        
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    @task
    def train_model(split, model_id_name, gcp_info):
        X_train, y_train = split['X_train'], split['y_train']
        # set time as index
        # X_train = X_train.set_index('time')

        # Set seed for reproducibility
        np.random.seed(0)
        tf.random.set_seed(0)

        # Define the model using the functional API
        input_shape = (X_train.shape[1],)
        input_layer = keras.Input(shape=input_shape)

        # Shared layers with L2 regularization
        x = layers.Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_layer)
        x = layers.BatchNormalization()(x)

        # Task-specific layers for temperature prediction with L2 regularization
        x_temp = layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        output_temp = layers.Dense(1, name='temperature')(x_temp)

        # Task-specific layers for precipitation prediction with L2 regularization
        x_precip = layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        output_precip = layers.Dense(1, name='precipitation')(x_precip)

        model = keras.Model(inputs=input_layer, outputs=[output_temp, output_precip])

        # Compile with added L2 loss and Adam optimizer with learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)

        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=['mae']
        )

        # Add Early Stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train with Early Stopping
        history = model.fit(
            X_train,
            {'temperature': y_train['temperature_2m_mean____C_'], 'precipitation': y_train['precipitation_sum__mm_']},
            validation_split=0.2,
            shuffle=False,
            epochs=100,  # More epochs with early stopping
            callbacks=[early_stopping]
        )

        # save model
        # model.save("model.h5") # save model to local storage
        # model.save(MODEL_PATH) # save model to GCS
        local_path = save_model_to_gcs(model, model_id_name, gcp_info)
        return local_path
    
    def save_model_to_gcs(model, model_id_name, gcp_info):
        BUCKET_NAME = gcp_info['bucket_name']
        local_path = './tmp/' + model_id_name + '.h5'
        model.save(local_path)

        # Upload the model to GCS
        from google.cloud import storage
        gcs_path = 'models/' + model_id_name + '.h5'
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        return local_path

    @task
    def predict_and_store(model_path, split, model_id_name, gcp_info):
        X_train, y_train = split['X_train'], split['y_train']
        X_test, y_test = split['X_test'], split['y_test']

        # get 1 month of data
        len_test = len(X_test)
        if len_test > 30:
            X_test = X_test[-30:]
            y_test = y_test[-30:]
        else:
            # also take some from train
            X_test = pd.concat([X_train[-(30-len_test):], X_test])
            y_test = pd.concat([y_train[-(30-len_test):], y_test])


        # Load the model
        model = keras.models.load_model(model_path)
        
        # Generate predictions
        predictions = model.predict(X_test)

        # Reshape or slice the predictions to make them 2-dimensional
        temp_predictions = predictions[0].flatten()  # Flatten the temperature predictions
        precip_predictions = predictions[1].flatten()  # Flatten the precipitation predictions

        # Create a DataFrame for the predictions
        predictions_df = pd.DataFrame({
            'temp_predict': temp_predictions,
            'precip_predict': precip_predictions
        })

        # Reset index of y_test to get the date column
        y_test.reset_index(inplace=True)

        # Combine the date and predictions
        predictions_df['weather_date'] = y_test['time']
        predictions_df['model_id_name'] = model_id_name

        # Reorder columns to match BigQuery table schema
        predictions_df = predictions_df[['model_id_name', 'weather_date', 'temp_predict', 'precip_predict']]

        # Get GCP info from Airflow variables
        PROJECT_ID = gcp_info['project_id']
        DATASET_ID = gcp_info['dataset_id']
        PREDICT_TABLE_ID = gcp_info['predict_table_id']
        
        # Create a BigQuery client
        client = bigquery.Client(project=PROJECT_ID)
        # Define the table to which to write
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{PREDICT_TABLE_ID}"
        # Define job configuration
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        # Write the DataFrame to BigQuery
        client.load_table_from_dataframe(predictions_df, table_ref, job_config=job_config).result()

    trigger_eval_dag = TriggerDagRunOperator(
        task_id="trigger_eval_dag",
        trigger_dag_id="eval_dag", 
        conf={'model_id_name': '{{ task_instance.xcom_pull(task_ids="set_model_id_name") }}'},
    )

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

    # Define the DAG flow
    model_id_name = set_model_id_name()
    data = fetch_weather_data(gcp_info)
    split_output = split_data(data)
    model_path = train_model(split_output, model_id_name, gcp_info)
    predict_and_store(
        model_path, 
        split_output, 
        model_id_name,
        gcp_info
    ) >> trigger_eval_dag

weather_prediction_training_dag = weather_prediction_training()
