import pandas as pd

import eda
import feature_engineering


if __name__ == '__main__':
    eda.concatenate_data()
    eda.clean_data()
    eda.create_outcome()
    eda.main_df.to_csv('project/main_df.csv')
    main_df = pd.read_csv('project/main_df.csv')
    feature_engineering.calculate_goals_sofar()
    feature_engineering.calculate_points_sofar()
    feature_engineering.calculate_form()
    feature_engineering.create_cleaned_dataset()
