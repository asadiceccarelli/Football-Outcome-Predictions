from distutils.command.upload import upload
import os
import logging
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv()


def rds_connect():
    """Connects to the AWS database using environment variables
    in a .env file.
    """
    DATABASE_TYPE = 'postgresql'
    DBAPI = 'psycopg2'
    HOST = os.environ['DB_HOST']
    USER = 'postgres'
    PASSWORD = os.environ['DB_PASSWORD']
    DATABASE = 'football-predictions'
    PORT = 5432
    return create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")


def upload_initial_data():
    """Uploads the initial dataframe into an RDS database in the cloud."""
    logging.info('Connecting to RDS...')
    engine = rds_connect()
    logging.info('Uploading initial dataset to RDS...')
    clean_dataset = pd.read_csv('preparation/dataframes/cleaned_dataset.csv', index_col=0)
    df_name = 'match-results'
    clean_dataset.to_sql(df_name, engine, if_exists='replace', index=False)


def upload_additional_data():
    logging.info('Connecting to RDS...')
    engine = rds_connect()
    logging.info('Uploading additional dataset to RDS...')
    clean_dataset_additional = pd.read_csv('project/dataframes/cleaned_dataset.csv', index_col=0)
    df_name = 'match-results'
    clean_dataset_additional.to_sql(df_name, engine, if_exists='append', index=False)
