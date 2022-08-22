import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)


def calculate_goals_sofar(df):
    """Iterate through the dataframe and insert the goals scored/conceeded
    so far over the course of the season into the main dataframe. Saved as a .csv file.
    Args:
        df (DataFrame): DataFrame to be calculated from.
    Returns:
        df (DataFrame): DataFrame with goals_sofar columns for home and away.
    """
    for league in df.league.unique():
        all_seasons_df = df[df.league == 'premier_league']
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {league} {i}...')
            print(f'Iterating through {league} {i}...')
            for team in df[(df.season == i) & (df.league == league)].home_team.unique():
                season_length = len(df.loc[
                        (df['league'] == league)
                        & (df['season'] == i)
                        & ((df['home_team'] == team) | (df['away_team'] == team)), 'round'])
                df.loc[
                        (df['league'] == league)
                        & (df['season'] == i)
                        & ((df['home_team'] == team) | (df['away_team'] == team)), 'round'
                        ] = [x for x in range(1, season_length + 1)]  # Renumbers rounds in case of double gameweeks. Possibly move to data-cleaning

                season_df = df[(df['season'] == i) & (df.league == league) & ((df.home_team == team) | (df.away_team == team))]
                scored_list = [0]
                conceeded_list = [0]
                for j in season_df['round'].unique():
                    match = season_df.loc[(season_df['round'] == j) & ((season_df['home_team'] == team) | (season_df['away_team'] == team))]
                    if match.home_team.item() == team:
                        scored_list.append(match.home_goals.item())
                        conceeded_list.append(match.away_goals.item())
                        scored_list[j] += scored_list[j-1]
                        conceeded_list[j] += conceeded_list[j-1]
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'home_scored_sofar'
                            ] = scored_list[j]
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'home_conceeded_sofar'
                            ] = conceeded_list[j]
                    else:
                        scored_list.append(match.away_goals.item())
                        conceeded_list.append(match.home_goals.item())
                        scored_list[j] += scored_list[j-1]
                        conceeded_list[j] += conceeded_list[j-1]
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'away_scored_sofar'
                            ] = scored_list[j]
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'away_conceeded_sofar'
                            ] = conceeded_list[j]
    return df
    

def calculate_points_sofar(df):
    """Iterate through dataframe and create new column with the points
    accumulated over the season so far. Saved as a .csv file.
    Args:
        df (DataFrame): DataFrame to be calculated from.
    Returns:
        df (DataFrame): DataFrame with points_sofar columns for home and away.
    """
    for league in df.league.unique():
        all_seasons_df = df[df.league == 'premier_league']
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {league} {i}...')
            for team in df[(df.season == i) & (df.league == league)].home_team.unique():
                season_length = len(df.loc[
                        (df['league'] == league)
                        & (df['season'] == i)
                        & ((df['home_team'] == team) | (df['away_team'] == team)), 'round'])
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
                            ] = points_sofar_list[j]
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
                            ] = points_sofar_list[j]
    return df


def calculate_form(df):
    """Iterate through dataframe and create new column with form over the past
    5 games of the season as a string object, e.g. 'WWDLD'. Saved as a .csv
    file.
    Args:
        df (DataFrame): DataFrame to be calculated from.
    Returns:
        df (DataFrame): DataFrame with form columns for home and away.
    """
    for league in df.league.unique():
        all_seasons_df = df[df.league == 'premier_league']
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {league} {i}...')
            print(f'Iterating through {league} {i}...')
            for team in df[(df.season == i) & (df.league == league)].home_team.unique():
                season_length = len(df.loc[
                        (df['league'] == league)
                        & (df['season'] == i)
                        & ((df['home_team'] == team) | (df['away_team'] == team)), 'round'])
                df.loc[
                        (df['league'] == league)
                        & (df['season'] == i)
                        & ((df['home_team'] == team) | (df['away_team'] == team)), 'round'
                        ] = [x for x in range(1, season_length + 1)]  # Renumbers rounds in case of double gameweeks. Possibly move to data-cleaning

                season_df = df[
                    (df['season'] == i)
                    & (df.league == league)
                    & ((df.home_team == team) | (df.away_team == team))]
                form = '-----'
                for j in season_df['round'].unique():
                    match = season_df.loc[
                        (season_df['round'] == j)
                        & ((season_df['home_team'] == team) | (season_df['away_team'] == team))]
                    if match.home_team.item() == team:
                        if match.outcome.item() == 1:
                            form = form[1:] + 'W'
                        elif match.outcome.item() == 0:
                            form = form[1:] + 'D'
                        else:
                            form = form[1:] + 'L'
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'home_form'
                            ] = form
                    else:
                        if match.outcome.item() == 1:
                            form = form[1:] + 'W'
                        elif match.outcome.item() == 0:
                            form = form[1:] + 'D'
                        else:
                            form = form[1:] + 'L'
                        df.loc[
                            (df['league'] == league)
                            & (df['season'] == i)
                            & (df['round'] == j)
                            & ((df['home_team'] == team) | (df['away_team'] == team)), 'away_form'
                            ] = form
    return df


def create_cleaned_dataset(df):
    """Calculate goals, points and form so far for home and away teams.
    Args:
        df (DataFrame): Clean dataframe with all data.
    Returns:
        cleaned_dataset (DataFrame): DataFrame with only relevant columns
            reading for modelling.
    """
    goals_sofar = calculate_goals_sofar(df)
    points_sofar = calculate_points_sofar(df)[['home_points_sofar', 'away_points_sofar']]
    form = calculate_form(df)[['home_form', 'away_form']]
    goals_sofar = goals_sofar[[
        'home_team', 'away_team', 'season', 'round', 'league', 'elo_home',
        'elo_away', 'home_goals', 'away_goals', 'outcome', 'home_scored_sofar',
        'home_conceeded_sofar', 'away_scored_sofar', 'away_conceeded_sofar']]
    cleaned_dataset = goals_sofar.join(points_sofar).join(form)  # Join on index
    return cleaned_dataset
