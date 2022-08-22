from eda import perform_eda
from feature_engineering import create_cleaned_dataset
from rds import upload_initial_data, upload_additional_data


def initial_data_pipeline():
    """Perfoms EDA, calculates goals, points and form so far for
    home and away teams and uploads DataFrame to a database in 
    the cloud.
    """
    main_df = perform_eda('Football-Dataset/*/*', 'Additional-Data/elo_dict.pkl')
    create_cleaned_dataset(main_df).to_csv('project/dataframes/cleaned_dataset.csv')
    upload_initial_data()


def upload_new_data(dataset_path, elo_path):
    """Perfoms EDA, calculates goals, points and form so far for
    home and away teams from new data and appends DataFrame to the
    pre-existing database in the cloud.
    """
    main_df_additional = perform_eda(dataset_path, elo_path, 'additional')
    create_cleaned_dataset(main_df_additional).to_csv('project/dataframes/cleaned_dataset_additional.csv')
    upload_additional_data()
