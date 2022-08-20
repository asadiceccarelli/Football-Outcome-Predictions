import logging
import pandas as pd
from eda import get_main_df

# main_df = get_main_df()
main_df = pd.read_csv('project/main_df.csv')

def calculate_goals_sofar():
    """Iterate through the dataframe and insert the goals scored/conceeded
    so far over the course of the season into the main dataframe. Saved as a .csv file.
    """
    for league in main_df.league.unique():
        all_seasons_df = main_df[main_df.league == 'premier_league']
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {league} {i}...')
            print(f'Iterating through {league} {i}...')
            for team in main_df[(main_df.season == i) & (main_df.league == league)].home_team.unique():
                season_length = len(main_df.loc[
                        (main_df['league'] == league)
                        & (main_df['season'] == i)
                        & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ])
                main_df.loc[
                        (main_df['league'] == league)
                        & (main_df['season'] == i)
                        & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ] = [x for x in range(1, season_length + 1)]  # Renumbers rounds in case of double gameweeks. Possibly move to data-cleaning

                season_df = main_df[(main_df['season'] == i) & (main_df.league == league) & ((main_df.home_team == team) | (main_df.away_team == team))]
                scored_list = [0]
                conceeded_list = [0]
                for j in season_df['round'].unique():
                    match = season_df.loc[(season_df['round'] == j) & ((season_df['home_team'] == team) | (season_df['away_team'] == team))]
                    if match.home_team.item() == team:
                        scored_list.append(match.home_goals.item())
                        conceeded_list.append(match.away_goals.item())
                        scored_list[j] += scored_list[j-1]
                        conceeded_list[j] += conceeded_list[j-1]
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'home_scored_sofar'
                            ] = scored_list[j]
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'home_conceeded_sofar'
                            ] = conceeded_list[j]
                    else:
                        scored_list.append(match.away_goals.item())
                        conceeded_list.append(match.home_goals.item())
                        scored_list[j] += scored_list[j-1]
                        conceeded_list[j] += conceeded_list[j-1]
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'away_scored_sofar'
                            ] = scored_list[j]
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'away_conceeded_sofar'
                            ] = conceeded_list[j]
    
    main_df.to_csv('project/main_df_goals_sofar.csv')
    

def calculate_points_sofar():
    """Iterate through dataframe and create new column with the points
    accumulated over the season so far. Saved as a .csv file.
    """
    for league in main_df.league.unique():
        all_seasons_df = main_df[main_df.league == 'premier_league']
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {league} {i}...')
            for team in main_df[(main_df.season == i) & (main_df.league == league)].home_team.unique():
                season_length = len(main_df.loc[
                        (main_df['league'] == league)
                        & (main_df['season'] == i)
                        & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ])
                main_df.loc[
                        (main_df['league'] == league)
                        & (main_df['season'] == i)
                        & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ] = [x for x in range(1, season_length + 1)]  # Renumbers rounds in case of double gameweeks. Possibly move to data-cleaning

                season_df = main_df[
                    (main_df['season'] == i)
                    & (main_df.league == league)
                    & ((main_df.home_team == team) | (main_df.away_team == team))]
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
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'home_points_sofar'
                            ] = points_sofar_list[j]
                    else:
                        if match.outcome.item() == 1:
                            points_sofar_list.append(0)
                        elif match.outcome.item() == 0:
                            points_sofar_list.append(1)
                        else:
                            points_sofar_list.append(3)
                        points_sofar_list[j] += points_sofar_list[j-1]
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'away_points_sofar'
                            ] = points_sofar_list[j]

    main_df.to_csv('project/main_df_points_sofar.csv')


def calculate_form():
    """Iterate through dataframe and create new column with form over the past
    5 games of the season as a string object, e.g. 'WWDLD'. Saved as a .csv
    file.
    """
    for league in main_df.league.unique():
        all_seasons_df = main_df[main_df.league == 'premier_league']
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {league} {i}...')
            print(f'Iterating through {league} {i}...')
            for team in main_df[(main_df.season == i) & (main_df.league == league)].home_team.unique():
                season_length = len(main_df.loc[
                        (main_df['league'] == league)
                        & (main_df['season'] == i)
                        & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ])
                main_df.loc[
                        (main_df['league'] == league)
                        & (main_df['season'] == i)
                        & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ] = [x for x in range(1, season_length + 1)]  # Renumbers rounds in case of double gameweeks. Possibly move to data-cleaning

                season_df = main_df[
                    (main_df['season'] == i)
                    & (main_df.league == league)
                    & ((main_df.home_team == team) | (main_df.away_team == team))]
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
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'home_form'
                            ] = form
                    else:
                        if match.outcome.item() == 1:
                            form = form[1:] + 'W'
                        elif match.outcome.item() == 0:
                            form = form[1:] + 'D'
                        else:
                            form = form[1:] + 'L'
                        main_df.loc[
                            (main_df['league'] == league)
                            & (main_df['season'] == i)
                            & (main_df['round'] == j)
                            & ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'away_form'
                            ] = form

    main_df.to_csv('project/main_df_form.csv')

calculate_form()
