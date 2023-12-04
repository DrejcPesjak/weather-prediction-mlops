import datetime
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
# from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
# from airflow.providers.google.cloud.sensors.gcs import GCSObjectsWithPrefixExistenceSensor
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from google.cloud import bigquery

# Constants and configurations
PROJECT_ID = 'balmy-apogee-404909'
BUCKET_NAME = 'europe-central2-rso-ml-airf-05c3abe0-bucket'
DATASET_ID = 'weather_prediction'
WEATHER_TABLE_ID = 'weather_history_LJ'
PREDICT_TABLE_ID = 'weather_predictions'
MODEL_NAME = 'dnn_multitarget'
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_PATH = f"gs://{BUCKET_NAME}/models/{MODEL_NAME}_{TIMESTAMP}.h5"

default_args = {
    'dag_id': 'weather_prediction_training',
    'owner': 'user',
    'start_date': datetime.datetime(2023, 1, 1),
    'email': ['your-email@example.com'],
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
    def fetch_weather_data():
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT * 
            FROM `{PROJECT_ID}.{DATASET_ID}.{WEATHER_TABLE_ID}`
        """
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df

    @task
    def split_data(df: pd.DataFrame):
        # Assuming preprocessing in ETL pipeline

        # Split the data, last 10 rows as test set
        test_size = 10
        X_train, X_test = df.iloc[:-test_size], df.iloc[-test_size:]
        y_train = X_train[['temperature_2m_mean____C_', 'precipitation_sum__mm_']]
        y_test = X_test[['temperature_2m_mean____C_', 'precipitation_sum__mm_']]
        
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    @task
    def train_model(split):
        X_train, y_train = split['X_train'], split['y_train']
        # set time as index
        X_train = X_train.set_index('time')

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

        return model


    @task
    def save_model_to_gcs(model):
        model.save(MODEL_PATH)

    @task
    def predict_and_store(model, split, project_id, dataset_id, table_id, model_id_name):
        X_test, y_test = ['X_test'], split['y_test']
        # set time as index
        X_test = X_test.set_index('time')
        
        # Generate predictions
        predictions = model.predict(X_test)

        # Create a DataFrame for the predictions
        predictions_df = pd.DataFrame(predictions, columns=['temp_predict', 'precip_predict'])
        # Reset index of y_test to get the date column
        y_test.reset_index(inplace=True)
        # Combine the date and predictions
        predictions_df['weather_date'] = y_test['time']
        predictions_df['model_id_name'] = model_id_name
        # Reorder columns to match BigQuery table schema
        predictions_df = predictions_df[['model_id_name', 'weather_date', 'temp_predict', 'precip_predict']]

        # Create a BigQuery client
        client = bigquery.Client(project=project_id)
        # Define the table to which to write
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        # Define job configuration
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        # Write the DataFrame to BigQuery
        client.load_table_from_dataframe(predictions_df, table_ref, job_config=job_config).result()

    trigger_eval_dag = TriggerDagRunOperator(
        task_id="trigger_eval_dag",
        trigger_dag_id="eval_dag", 
        conf={'model_id_name': MODEL_NAME + '_' + TIMESTAMP},
    )

    # Define DAG flow
    data = fetch_weather_data()
    split_output = split_data(data)
    model = train_model(split_output)
    save_model_to_gcs(model)
    predict_and_store(
        model, 
        split_output,
        PROJECT_ID, 
        DATASET_ID, 
        PREDICT_TABLE_ID, 
        MODEL_NAME + '_' + TIMESTAMP
    ) >> trigger_eval_dag

weather_prediction_training_dag = weather_prediction_training()
