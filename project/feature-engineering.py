import logging
import pandas as pd
from eda import get_main_df

# main_df = get_main_df()
main_df = pd.read_csv('main_df.csv')

def goals_so_far():
    """Iterate through the entire dataframe and insert the goals scored/conceeded so far by each time over the course of the season into the main dataframe."""
    for league in main_df.league.unique():
        all_seasons_df = main_df[main_df.league == 'premier_league']
        for i in all_seasons_df.season.unique():
            logging.info(f'Iterating through {i} of {league}.')
            for team in main_df[(main_df.season == i) & (main_df.league == league)].home_team.unique():
                season_length = len(main_df.loc[
                        (main_df['league'] == league) &
                        (main_df['season'] == i) &
                        ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ])
                main_df.loc[
                        (main_df['league'] == league) &
                        (main_df['season'] == i) &
                        ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'round'
                        ] = [x for x in range(1, season_length + 1)]  # Renumbers rounds in case of double gameweeks. Possibly move to data-cleaning

                season_df = main_df[(main_df['season'] == i) & (main_df.league == league) & ((main_df.home_team == team) | (main_df.away_team == team))]
                scored_sofar_dict = {}
                conceeded_sofar_dict = {}
                scored_sofar_dict[team] = [0]
                conceeded_sofar_dict[team] = [0]
                for j in season_df['round'].unique():
                    match = season_df.loc[(season_df['round'] == j) & ((season_df['home_team'] == team) | (season_df['away_team'] == team))]
                    if match.home_team.item() == team:
                        scored_sofar_dict[team].append(match.home_goals.item())
                        conceeded_sofar_dict[team].append(match.away_goals.item())
                        scored_sofar_dict[team][j] += scored_sofar_dict[team][j-1]
                        conceeded_sofar_dict[team][j] += conceeded_sofar_dict[team][j-1]
                        main_df.loc[
                            (main_df['league'] == league) &
                            (main_df['season'] == i) &
                            (main_df['round'] == j) &
                            ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'home_scored_sofar'
                            ] = scored_sofar_dict[team][j]
                        main_df.loc[
                        (main_df['league'] == league) &
                        (main_df['season'] == i) &
                        (main_df['round'] == j) &
                        ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'home_conceeded_sofar'
                        ] = conceeded_sofar_dict[team][j]
                    else:
                        scored_sofar_dict[team].append(match.away_goals.item())
                        conceeded_sofar_dict[team].append(match.home_goals.item())
                        scored_sofar_dict[team][j] += scored_sofar_dict[team][j-1]
                        conceeded_sofar_dict[team][j] += conceeded_sofar_dict[team][j-1]
                        main_df.loc[
                            (main_df['league'] == league) &
                            (main_df['season'] == i) &
                            (main_df['round'] == j) &
                            ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'away_scored_sofar'
                            ] = scored_sofar_dict[team][j]
                        main_df.loc[
                        (main_df['league'] == league) &
                        (main_df['season'] == i) &
                        (main_df['round'] == j) &
                        ((main_df['home_team'] == team) | (main_df['away_team'] == team)), 'away_conceeded_sofar'
                        ] = conceeded_sofar_dict[team][j]
    return main_df

