import datetime
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import requests
from io import StringIO
import pandas as pd
from google.cloud import bigquery

daily_params = ','.join([
            'weather_code',
            'temperature_2m_max',
            'temperature_2m_min',
            'temperature_2m_mean',
            'apparent_temperature_max',
            'apparent_temperature_min',
            'apparent_temperature_mean',
            'sunrise',
            'sunset',
            # 'daylight_duration',
            # 'sunshine_duration',
            'precipitation_sum',
            'rain_sum',
            'snowfall_sum',
            'precipitation_hours',
            'wind_speed_10m_max',
            'wind_gusts_10m_max',
            'wind_direction_10m_dominant',
            'shortwave_radiation_sum',
            'et0_fao_evapotranspiration'
        ])


def get_batch_daily_weather_csv(lat, lon, start_date, end_date):
    # Getting the date 10 years ago from today
    # start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    if start_date is None:
        start_date = '2000-01-01'
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    endpoint = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': daily_params,
        'timezone': 'auto',
        'format': 'csv'
    }
    
    response = requests.get(endpoint, params=params)
    
    if response.status_code == 200:
        data = response.text
        return data
    else:
        print(f'Failed to retrieve data: {response.status_code}')
        return None


def preprocess_data(df_orig):
    df = df_orig.copy()

    # drop rows with NaN values
    df = df.dropna()

    # First, perform date conversions before setting 'time' as index
    df['time'] = pd.to_datetime(df['time'])
    df['year_since_2000'] = df['time'].dt.year - 2000
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['day_of_year'] = df['time'].dt.dayofyear
    df['weekday'] = df['time'].dt.weekday

    # extract hour in day from sunrise and sunset columns
    df['sunrise'] = pd.to_datetime(df['sunrise (iso8601)'])
    df['sunrise'] = df['sunrise'].dt.hour + df['sunrise'].dt.minute / 60
    df['sunset'] = pd.to_datetime(df['sunset (iso8601)'])
    df['sunset'] = df['sunset'].dt.hour + df['sunset'].dt.minute / 60

    # Now set 'time' as index
    # df = df.set_index('time')

    # Drop columns after extracting necessary features
    df = df.drop(columns=['sunrise (iso8601)', 'sunset (iso8601)'])

    # Prepare target variables y and shift them by 1 day
    y = df[['temperature_2m_mean (°C)', 'precipitation_sum (mm)']].shift(-1)
    y = y.dropna()

    # Since we shifted y, we need to drop the last row of df to align it with y
    df = df.iloc[:-1]

    # missing wind gust
    df.insert(loc=13, column='wind_gusts_10m_max (km/h)', value=df['wind_speed_10m_max (km/h)'])

    return df, y

default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'email': ['drejcpesjak.pesjak@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=2)
}

@dag(
    dag_id='etl_dag',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=days_ago(2),
    tags=['open_meteo', 'download', 'bigquery'],
    catchup=False 
)
def open_meteo_download():
    @task
    def extract(lat, lon, bigquery_info):
        project_id = bigquery_info['project_id']
        dataset_id = bigquery_info['dataset_id']
        table_id = bigquery_info['table_id']

        # Set up BigQuery client
        client = bigquery.Client(project=project_id)

        # Fetch the latest date from BigQuery
        query = f"""
            SELECT * 
            FROM `{project_id}.{dataset_id}.{table_id}` 
            ORDER BY time DESC 
            LIMIT 1
        """
        res = client.query(query).to_dataframe()
        latest_date = res['time'].astype(str)[0]
        print(f'Latest date in BigQuery: {latest_date}')
        colnames = res.columns.tolist()

        # Fetch new data from OpenMeteo
        new_batch_csv = get_batch_daily_weather_csv(lat, lon, latest_date, None)
        new_batch_csv_io = StringIO(new_batch_csv)
        new_df = pd.read_csv(new_batch_csv_io, skiprows=3)
        print(f'New data shape: {new_df.shape}')

        return {'new_df': new_df, 'colnames': colnames}

    @task
    def transform(data):
        new_df, colnames = data['new_df'], data['colnames']
        proc_df, proc_y = preprocess_data(new_df)
        proc_df.columns = colnames
        return proc_df

    @task()
    def load(proc_df, bigquery_info):
        project_id = bigquery_info['project_id']
        dataset_id = bigquery_info['dataset_id']
        table_id = bigquery_info['table_id']

        # Set up BigQuery client
        client = bigquery.Client(project=project_id)

        # Configure the load job
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND"
        )

        # Load the DataFrame to BigQuery
        job = client.load_table_from_dataframe(
            proc_df, 
            f'{project_id}.{dataset_id}.{table_id}',
            job_config=job_config
        )

        job.result()
        print(f'BigQuery load job {job.job_id} is complete, with state: {job.state}')
        print(f'Loaded {job.output_rows} rows to {project_id}.{dataset_id}.{table_id}')

    # Define the DAG flow
    bigquery_info = {
        'project_id': 'balmy-apogee-404909',
        'dataset_id': 'weather_prediction',
        'table_id': 'weather_history_LJ'
    }
    extract_ouput = extract(46.0511, 14.5051, bigquery_info)
    processed_data = transform(extract_ouput)
    load(processed_data, bigquery_info)

open_meteo_download_dag = open_meteo_download()
