import os
import logging
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv()


def rds_connect():
    DATABASE_TYPE = 'postgresql'
    DBAPI = 'psycopg2'
    HOST = os.environ['DB_HOST']
    USER = 'postgres'
    PASSWORD = os.environ['DB_PASSWORD']
    DATABASE = 'football-predictions'
    PORT = 5432
    return create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")


logging.info('Connecting to RDS...')
engine = rds_connect()

logging.info('Uploading to RDS...')
clean_dataset = pd.read_csv('project/dataframes/cleaned_dataset.csv')
df_name = 'match-results'
clean_dataset.to_sql(df_name, engine, if_exists='replace', index=False)

