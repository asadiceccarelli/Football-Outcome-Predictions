import glob
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt


def concatenate_data():
    '''Combine all dataframes into one singular dataframe to be inspected.'''
    global match_df
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
    
    raw_df['clean_link'] = raw_df['Link'].apply(clean_link)
    match_info_df = pd.read_csv('https://aicore-files.s3.amazonaws.com/Data-Science/Match_Info.csv')
    match_df = pd.merge(raw_df, match_info_df, left_on='clean_link', right_on='Link')
    msno.matrix(match_df)



def clean_data(df):
    df.columns= df.columns.str.lower()
    match_df.drop(['link_x', 'link_y'], axis=1, inplace=True)
    df.drop_duplicates('clean_link', inplace=True)

    def clean_referee(ref):
        if isinstance(ref, str) == True:
            new_ref = ref.split('\r\n')[1][9:]
            return new_ref

    df['referee'] = df['referee'].apply(clean_referee)


def create_outcome(df):
    '''Creates an outcome column where a home win == 1, draw == 0 and away win == -1. Drops uneccessary link and result columns.'''
    df.drop(df[df['Result'].str.len() != 3].index, inplace=True)  # Remove any scores set as dates
    df['Home_Goals'] = df['Result'].apply(lambda x: x.split('-')[0])
    df['Away_Goals'] = df['Result'].apply(lambda x: x.split('-')[-1])
    df['Outcome'] = df[['Home_Goals', 'Away_Goals']].apply(
        lambda x: 1 if x['Home_Goals'] > x['Away_Goals'] else (-1 if x['Home_Goals'] < x['Away_Goals'] else 0), axis=1)
    df.drop('Link', axis=1, inplace=True)


def plot_outcome(df):
    '''Plots the outcome of all leagues against time and saves the figure to README-images.'''
    plt.figure(0)
    df.groupby('Season')['Outcome'].mean().plot()
    plt.xlabel('Year')
    plt.ylabel('Result')
    plt.savefig('README-images/outcome-over-time.jpg', bbox_inches='tight', dpi=300)


def plot_goals(df):
    '''Removes matches where scores not set as integers.
    Plots the average goals per game of each league and saves the figure to README-images.'''
    df.drop(df[df['Result'].str.len() != 3].index, inplace=True)  # Remove any scores set as dates
    df['Home_Goals'] = df['Home_Goals'].astype('int64')
    df['Away_Goals'] = df['Away_Goals'].astype('int64')
    df['Total_Goals'] = df[['Home_Goals','Away_Goals']].apply(lambda x: x['Home_Goals'] + x['Away_Goals'], axis=1)

    plt.figure(1)
    df.groupby('League')['Total_Goals'].mean().plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Goals')
    plt.savefig('README-images/average-goals.jpg', bbox_inches='tight', dpi=300)


def plot_rounds(df):
    '''Plots the number of rounds in each league against time and saves the figure to README-images.'''
    plt.figure(2)
    plt.tight_layout(pad=5)
    df_leagues = df['League'].unique()
    for league in df_leagues:
        df[df['League'] == f'{league}'].groupby('Season')['Round'].max().plot()
    plt.xlabel('Year')
    plt.ylabel('Number of Rounds')
    plt.legend(df_leagues, bbox_to_anchor =(1, 1.03))
    plt.savefig('README-images/rounds.jpg', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    concatenate_data()
    # create_outcome(df)
    # plot_outcome(df)
    # plot_goals(df)
    # plot_rounds(df)