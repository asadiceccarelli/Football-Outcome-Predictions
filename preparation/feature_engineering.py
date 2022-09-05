import logging

logging.basicConfig(level=logging.INFO)


def calculate_average_goals(df):
    """Iterate through the dataframe and insert the average goals scored/conceeded
    over the club's past 10 matches into the main dataframe.
    Args:
        df (DataFrame): DataFrame to be calculated from.
    Returns:
        df (DataFrame): DataFrame with average_recent_scored/conceeded columns for home and away.
    """
    df.sort_values('date_new')
    for team in df.home_team.unique():
        logging.info(f"Calculating {team}'s recent goals...")
        goals_scored = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        goals_conceeded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for index, match in df.iterrows():
            if match.home_team == team:
                df.loc[index, 'average_recent_home_scored'] = sum(goals_scored) / len(goals_scored)
                df.loc[index, 'average_recent_home_conceeded'] = sum(goals_conceeded) / len(goals_conceeded)
                goals_scored = goals_scored[1:] + [match.home_goals]
                goals_conceeded = goals_conceeded[1:] + [match.away_goals]
            elif match.away_team == team:
                df.loc[index, 'average_recent_away_scored'] = sum(goals_scored) / len(goals_scored)
                df.loc[index, 'average_recent_away_conceeded'] = sum(goals_conceeded) / len(goals_conceeded)
                goals_scored = goals_scored[1:] + [match.away_goals]
                goals_conceeded = goals_conceeded[1:] + [match.home_goals]
    return df
    

def calculate_points_sofar(df):
    """Iterate through dataframe and create new column with the points
    accumulated over the season so far.
    Args:
        df (DataFrame): DataFrame to be calculated from.
    Returns:
        df (DataFrame): DataFrame with points_sofar columns for home and away.
    """
    df.sort_values('date_new')
    for league in df.league.unique():
        all_seasons_df = df[df.league == league]
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {league} {i}...')
            for team in df[(df.season == i) & (df.league == league)].home_team.unique():
                season_length = len(df.loc[
                        (df['league'] == league)
                        & (df['season'] == i)
                        & ((df['home_team'] == team) | (df['away_team'] == team)), 'round'
                        ])
                df.loc[
                        (df['league'] == league)
                        & (df['season'] == i)
                        & ((df['home_team'] == team) | (df['away_team'] == team)), 'round'
                        ] = [x for x in range(1, season_length + 1)]  # Renumbers rounds in case of double gameweeks. Possibly move to data-cleaning

                season_df = df[
                    (df['season'] == i)
                    & (df.league == league)
                    & ((df.home_team == team) | (df.away_team == team))]
                points_sofar_list = [0]
                for j in season_df['round'].unique():
                    match = season_df.loc[
                        (season_df['round'] == j)
                        & ((season_df['home_team'] == team) | (season_df['away_team'] == team))]
                    if match.home_team.item() == team:
                        if match.outcome.item() == 1:
                            points_sofar_list.append(3)
                        elif match.outcome.item() == 0:
                            points_sofar_list.append(1)
                        else:
                            points_sofar_list.append(0)
                        points_sofar_list[j] += points_sofar_list[j-1]
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'home_points_sofar'
                            ] = points_sofar_list[j-1]
                    else:
                        if match.outcome.item() == 1:
                            points_sofar_list.append(0)
                        elif match.outcome.item() == 0:
                            points_sofar_list.append(1)
                        else:
                            points_sofar_list.append(3)
                        points_sofar_list[j] += points_sofar_list[j-1]
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'away_points_sofar'
                            ] = points_sofar_list[j-1]
    return df


def calculate_form(df):
    """Iterate through dataframe and create new column with form over the past
    5 games by calculating the sum of the last 5 outcomes.
    Args:
        df (DataFrame): DataFrame to be calculated from.
    Returns:
        df (DataFrame): DataFrame with form columns for home and away.
    """
    for team in df.home_team.unique():
        logging.info(f"Calculating {team}'s form...")
        form = [0, 0, 0, 0, 0]
        for index, match in df.iterrows():
            if match.home_team == team:
                df.loc[index, 'home_form'] = sum(form)
                if match.outcome == 1:
                    form = form[1:] + [1]
                elif match.outcome == -1:
                    form = form[1:] + [-1]
                else:
                    form = form[1:] + [0]
            elif match.away_team == team:
                df.loc[index, 'away_form'] = sum(form)
                if match.outcome == 1:
                    form = form[1:] + [-1]
                elif match.outcome == -1:
                    form = form[1:] + [1]
                else:
                    form = form[1:] + [0]
    return df


def create_cleaned_dataset(df):
    """Calculate goals, points and form so far for home and away teams.
    Args:
        df (DataFrame): Clean dataframe with all data.
    Returns:
        cleaned_dataset (DataFrame): DataFrame with only relevant columns
            reading for modelling.
    """
    goals_sofar = calculate_average_goals(df)
    points_sofar = calculate_points_sofar(df)[['home_points_sofar', 'away_points_sofar']]
    form = calculate_form(df)[['home_form', 'away_form']]
    goals_sofar = goals_sofar[[
        'round', 'elo_home', 'elo_away', 'outcome', 'average_recent_home_scored',
        'average_recent_home_conceeded',  'average_recent_away_scored',  'average_recent_away_conceeded'
        ]]
    cleaned_dataset = goals_sofar.join(points_sofar).join(form)  # Join on index
    return cleaned_dataset


import pandas as pd
goals_sofar = pd.read_csv('preparation/dataframes/main_df_average_goals.csv')
points_sofar = pd.read_csv('preparation/dataframes/main_df_points_sofar.csv')[['home_points_sofar', 'away_points_sofar']]
form = pd.read_csv('preparation/dataframes/main_df_form.csv')[['home_form', 'away_form']]
goals_sofar = goals_sofar[[
    'elo_home', 'elo_away', 'outcome', 'average_recent_home_scored',
    'average_recent_home_conceeded',  'average_recent_away_scored',  'average_recent_away_conceeded'
    ]]
cleaned_dataset = goals_sofar.join(points_sofar).join(form).to_csv('preparation/dataframes/cleaned_dataset.csv')  # Join on index