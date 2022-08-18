import logging
import glob
import pickle
import pandas as pd
import plotly.express as px


def concatenate_data():
    """Combine all dataframes into one singular dataframe, main_df."""
    global main_df
    logging.info('Concatenating data...')
    paths = glob.glob('Football-Dataset/*/*')
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
    match_info_df = pd.read_csv('https://aicore-files.s3.amazonaws.com/Data-Science/Match_Info.csv')
    team_info_df = pd.read_csv("https://aicore-files.s3.amazonaws.com/Data-Science/Team_Info.csv")
    elo_dict = pickle.load(open('elo_dict.pkl', 'rb'))
    elo_df = pd.DataFrame.from_dict(elo_dict)
    elo_df = elo_df.transpose().reset_index().rename(columns={'index': 'Link'})

    match_df = pd.merge(raw_df, match_info_df, left_on='clean_link', right_on='Link')
    mach_elo_df = pd.merge(match_df, elo_df, left_on='Link_x', right_on='Link', how='left')
    main_df = pd.merge(mach_elo_df, team_info_df, left_on='Home_Team', right_on='Team', how='left')


def clean_data():
    """Further cleans the dataframe by dropping unnecessary columns, removing duplicates and cleaning referee and capacity data."""
    logging.info('Cleaning data...')
    main_df.columns= main_df.columns.str.lower()
    main_df.drop(['link_x', 'link_y', 'team'], axis=1, inplace=True)
    main_df.drop_duplicates('clean_link', inplace=True)
    main_df.drop(main_df[main_df.league == 'eerste_divisie'].index, inplace=True)
    main_df.drop(main_df[main_df.league == 'segunda_liga'].index, inplace=True)
    main_df.drop(main_df[(main_df.league == 'championship') & (main_df['season'] == 1998)].index, inplace=True)  # Drop incomplete 1998 championship data
    main_df['referee'] = main_df.referee.apply(lambda x: x.split('\r\n')[1][9:] if type(x)==str else x)
    main_df['capacity'] = main_df.capacity.apply(lambda x: int(x.replace(',', '')) if type(x)==str else x)


def create_outcome():
    """Creates an outcome column where a home win == 1, draw == 0 and away win == -1. Drops uneccessary link and result columns."""
    main_df.drop(main_df[main_df.result.str.len() != 3].index, inplace=True)  # Remove any scores set as dates
    main_df['home_goals'] = main_df.result.apply(lambda x: x.split('-')[0])
    main_df['away_goals'] = main_df.result.apply(lambda x: x.split('-')[-1])
    main_df['home_goals'] = main_df.home_goals.astype('int64')
    main_df['away_goals'] = main_df.away_goals.astype('int64')
    main_df.drop('result', axis=1, inplace=True)
    main_df['outcome'] = main_df[['home_goals', 'away_goals']].apply(
        lambda x: 1 if x['home_goals'] > x['away_goals'] else (-1 if x['home_goals'] < x['away_goals'] else 0), axis=1)


def plot_outcome():
    """Plots the outcome of all leagues against time and saves the figure to README-images."""
    outcome_line = px.line(main_df.groupby('season')['outcome'].mean(), title='Outcome of matches over time')
    outcome_line.update_layout(showlegend=False)
    outcome_line.write_image('README-images/outcome-over-time.png')


def plot_goals():
    """ Plots the average goals per game of each league and saves the figure to README-images."""
    main_df['home_goals'] = main_df.home_goals.astype('int64')
    main_df['away_goals'] = main_df.away_goals.astype('int64')
    main_df['total_goals'] = main_df[['home_goals','away_goals']].apply(lambda x: x['home_goals'] + x['away_goals'], axis=1)
    goals_bar = px.bar(main_df.groupby('league')['total_goals'].mean(), title='Average number of goals per game')
    goals_bar.update_layout(showlegend=False)
    goals_bar.write_image('README-images/average_goals.png')


def plot_rounds():
    """Plots a line graph of the number of rounds in each league against time and saves the figure to README-images."""
    dict = {}
    leagues = main_df['league'].unique()
    for league in leagues:
        dict[league] = main_df[main_df.league == f'{league}'].groupby('season')['round'].max()
    rounds_line = px.line(
        dict, labels={'value': 'no. rounds'},
        title='Rounds per season for each league',
        color_discrete_sequence=px.colors.qualitative.Light24
        )
    rounds_line.write_image('README-images/rounds.png')

def plot_capacity():
    """Plots a scatter chart of stadium size vs the average home outcome of each game and saves the figure to README-images."""
    home_outcome = main_df.groupby('home_team')['outcome'].mean()
    capcity_outcome = pd.merge(home_outcome, main_df, left_on='home_team', right_on='home_team', how='left')
    capcity_outcome_scatter = px.scatter(
        x=capcity_outcome['capacity'],
        y=capcity_outcome['outcome_x'],
        trendline='ols',
        trendline_color_override='red',
        labels={'x': 'capacity', 'y': 'average outcome'},
        title='The effect of stadium size on the outcome of a match'
        )
    capcity_outcome_scatter.write_image('README-images/capacity.png')


def plot_cards():
    main_df['home_cards'] = main_df.apply(lambda x: x['home_yellow'] + x['home_red'], axis=1)
    home_cards = main_df.groupby('home_team')['home_cards'].mean()
    capacity_cards = pd.merge(home_cards, main_df, left_on='home_team', right_on='home_team', how='left')
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


def get_main_df():
    concatenate_data()
    clean_data()
    create_outcome()
    return main_df


if __name__ == '__main__':
    concatenate_data()
    clean_data()
    # plot_rounds()
    create_outcome()
    plot_outcome()
    plot_goals()
    plot_capacity()