import logging
import glob
import pickle
import pandas as pd
import plotly.express as px

logging.basicConfig(level=logging.INFO)


def concatenate_data(dataset_path, elo_path):
    """Combine all DataFrames into one singular DataFrame.
    Args:
        dataset_path (str): Relative path of .csv datasets.
        pickle_path (str): Relative path of .pkl file containing
            elo data.
    Returns:
        main_df (DataFrame): Contains a concatenation of all data."""
    logging.info('Concatenating data...')
    paths = glob.glob(dataset_path)
    pd_list = []
    for path in paths:
        temp_csv = pd.read_csv(path)
        pd_list.append(temp_csv)
    raw_df = pd.concat(pd_list)

    def clean_link(link):
        new_link = link.split('.com')[1]
        new_link_year = new_link.split('/')[-1][0:4]
        new_link_complete = '/'.join(new_link.split('/')[:-1]) + '/' + new_link_year
        return new_link_complete
    
    raw_df['clean_link'] = raw_df.Link.apply(clean_link)
    match_info_df = pd.read_csv('Additional-Data/match_info.csv', index_col=[0])
    team_info_df = pd.read_csv('Additional-Data/team_info.csv', index_col=[0])
    elo_dict = pickle.load(open(elo_path, 'rb'))  # Read binary file
    elo_df = pd.DataFrame.from_dict(elo_dict)
    elo_df = elo_df.transpose().reset_index().rename(columns={'index': 'Link'})

    match_df = pd.merge(raw_df, match_info_df, left_on='clean_link', right_on='Link')
    match_elo_df = pd.merge(match_df, elo_df, left_on='Link_x', right_on='Link', how='left')
    main_df = pd.merge(match_elo_df, team_info_df, left_on='Home_Team', right_on='Team', how='left')
    return main_df


def clean_data(df):
    """Further cleans the dataframe by dropping unnecessary columns,
    removing duplicates and cleaning referee and capacity data.
    Args:
        df (DataFrame): DataFrame to be cleaned.
    Returns:
        df (DataFrame): Clean DataFrame.
    """
    logging.info('Cleaning data...')
    df.columns = df.columns.str.lower()
    df.drop(['link_x', 'link_y', 'team'], axis=1, inplace=True)
    df.drop_duplicates('clean_link', inplace=True)
    df = df[df.elo_home.notna()]
    df = df[df.elo_away.notna()]
    df.drop(df[df.league == 'eerste_divisie'].index, inplace=True)
    df.drop(df[df.league == 'segunda_liga'].index, inplace=True)
    df.drop(df[(df.league == 'championship') & (df.season == 1998)].index, inplace=True)  # Drop incomplete data
    df.drop(df[(df.league == 'ligue_1') & (df.season == 1990)].index, inplace=True)
    df['referee'] = df.referee.apply(lambda x: x.split('\r\n')[1][9:] if type(x)==str else x)
    df['capacity'] = df.capacity.apply(lambda x: int(x.replace(',', '')) if type(x)==str else x)
    return df


def create_outcome(df):
    """Creates an outcome column where a home win == 1, draw == 0 and
    away win == -1. Drops uneccessary link and result columns.
    Args:
        df (DataFrame): DataFrame without outcome column.
    Returns:
        df (DataFrame): DataFrame with outcome calculated.
    """
    df.drop(df[df.result.str.len() != 3].index, inplace=True)  # Remove any scores incorrectly formatted
    df['home_goals'] = df.result.apply(lambda x: x.split('-')[0])
    df['away_goals'] = df.result.apply(lambda x: x.split('-')[-1])
    df['home_goals'] = df.home_goals.astype('int64')
    df['away_goals'] = df.away_goals.astype('int64')
    df.drop('result', axis=1, inplace=True)
    df['outcome'] = df[['home_goals', 'away_goals']].apply(
        lambda x: 1 if x['home_goals'] > x['away_goals'] else (-1 if x['home_goals'] < x['away_goals'] else 0), axis=1)
    return df


def plot_outcome(df):
    """Plots the outcome of all leagues against time and saves
    the figure to README-images.
    Args:
        df (DataFrame): DataFrame containing all data.
    """
    outcome_line = px.line(df.groupby('season')['outcome'].mean(), title='Outcome of matches over time')
    outcome_line.update_layout(showlegend=False)
    outcome_line.write_image('README-images/outcome-over-time.png')


def plot_goals(df):
    """ Plots the average goals per game of each league and saves
    the figure to README-images.
    Args:
        df (DataFrame): DataFrame containing all data.
    """
    df['home_goals'] = df.home_goals.astype('int64')
    df['away_goals'] = df.away_goals.astype('int64')
    df['total_goals'] = df[['home_goals','away_goals']].apply(lambda x: x['home_goals'] + x['away_goals'], axis=1)
    goals_bar = px.bar(df.groupby('league')['total_goals'].mean(), title='Average number of goals per game')
    goals_bar.update_layout(showlegend=False)
    goals_bar.write_image('README-images/average_goals.png')


def plot_rounds(df):
    """Plots a line graph of the number of rounds in each league
    against time and saves the figure to README-images.
    Args:
        df (DataFrame): DataFrame containing all data.
    """
    dict = {}
    leagues = df['league'].unique()
    for league in leagues:
        dict[league] = df[df.league == f'{league}'].groupby('season')['round'].max()
    rounds_line = px.line(
        dict, labels={'value': 'no. rounds'},
        title='Rounds per season for each league',
        color_discrete_sequence=px.colors.qualitative.Light24
        )
    rounds_line.write_image('README-images/rounds.png')

def plot_capacity(df):
    """Plots a scatter chart of stadium size vs the average home
    outcome of each game and saves the figure to README-images.
    Args:
        df (DataFrame): DataFrame containing all data.
    """
    home_outcome = df.groupby('home_team')['outcome'].mean()
    capcity_outcome = pd.merge(home_outcome, df, left_on='home_team', right_on='home_team', how='left')
    capcity_outcome_scatter = px.scatter(
        x=capcity_outcome['capacity'],
        y=capcity_outcome['outcome_x'],
        trendline='ols',
        trendline_color_override='red',
        labels={'x': 'capacity', 'y': 'average outcome'},
        title='The effect of stadium size on the outcome of a match'
        )
    capcity_outcome_scatter.write_image('README-images/capacity.png')


def plot_cards(df):
    """Plots a scatter chart of number of cards vs the average home
    outcome of each game and saves the figure to README-images.
    Args:
        df (DataFrame): DataFrame containing all data.
    """
    df['home_cards'] = df.apply(lambda x: x['home_yellow'] + x['home_red'], axis=1)
    home_cards = df.groupby('home_team')['home_cards'].mean()
    capacity_cards = pd.merge(home_cards, df, left_on='home_team', right_on='home_team', how='left')
    capacity_cards
    capacity_cards_scatter = px.scatter(
        x=capacity_cards['capacity'],
        y=capacity_cards['home_cards_x'],
        trendline='ols',
        trendline_color_override='red',
        labels={'x': 'capacity', 'y': 'average no. cards per game'},
        title='The effect of stadium size on the number of cards in a match'
        )
    capacity_cards_scatter.write_image('README-images/cards.png')


def perform_eda(dataset_path, pickle_path):
    """Concatenates, cleans and creates a column containing the match outcomes.
    Args:
        dataset_path (str): Relative path of .csv datasets.
        pickle_path (str): Relative path of .pkl file containing
            elo data.
    Returns:
        main_df_final (DataFrame): Cleaned DataFrame with all columns.
    """
    main_df = concatenate_data(dataset_path, pickle_path)
    main_df_clean = clean_data(main_df)
    main_df_final = create_outcome(main_df_clean)
    return main_df_final


if __name__ == '__main__':
    perform_eda('Football-Dataset/*/*', 'Additional-Data/elo_dict.pkl').to_csv('project/dataframes/main_df.csv')
    # # plot_rounds(main_df)
    # plot_outcome(main_df)
    # plot_goals(main_df)
    # plot_capacity(main_df)