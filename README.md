# Football Outcome Predictions
My forth and final project for AiCore. Building a machine learning model to make predictions for upcoming football matches.

## Milestone 1: Project Setup

- A new conda environment is set up with the name ```football-env``` so that the package requirements needed for the project can be written to ```requirements.txt``` with ease. Git branches will be used throughout the project following the Gitflow branching model.

<p align='center'>
  <img 
    width='500'
    src='README-images/gitflow.png'
  >
</p>

> The simplified Gitflow branching model.

## Milestone 2: Data Cleaning and EDA

Exploratory Data Analysis (EDA) is the first step that must be undertaken before creating any form of model. It involves validating that the data provided is clean and free of missing values so as not to cause problems when working with this large quanity of data later. By exploring this data, a rough understanding of the underlying trends between the variables can begin to be established.

### Data Cleaning

After concatenating the data provided into a singular dataframe ```main_df``` and running the ```describe()``` method on it, several key pieces of information show up:
- The data contains information from 142536 matches spanning 32 years across 14 different leagues.
- There is a discrepancy between the total number of links (142536) and the number of unique links (128379). Reasons for this could be some matches do not contain a link, contain the wrong link or there may be duplicates of matches in the dataframe.
- There is a discrepancy between the number of unique home teams (533) and the number of unique away teams (540). These too, should in theory have the same value.

Missing data can be visualised using the ```missingno``` package and running ```missingno.matix(main_df)```.

<p align='center'>
  <img 
    width='750'
    src='README-images/missingno.png'
  >
</p>

> Gaps in the plot show missing values such as ```NaN```.

The data is cleaned with a function perfoming the following:
- Duplicated rows with identical links removed with ```match_df.drop_duplicates('clean_link', inplace=True)```. Using ```main_df[main_df.duplicated('clean_link', keep=False)]```, it is revealed that the problem leagues containing the majority of duplicates are the Eerste Divisie and Segunda Liga, with a small minority from Ligue 1 and Serie B.
- The ```referee``` column is cleaned up to remove the ```\r\n``` on either side of the name. Care is taken when applying the ```clean_referee()``` function to the series to avoid any ```NaN``` floats. 
- The ```capacity``` column is cleaned and converted to an integer value where the value is not ```NaN```.

```python
df['referee'] = df['referee'].apply(lambda x: x.split('\r\n')[1][9:] if type(x)==str else x)
df['capacity'] = df['capacity'].apply(lambda x: int(x.replace(',', '')) if type(x)==str else x)
```
> Cleaning the ```referee``` and ```capacity``` column using ```lambda``` functions and the ```apply()``` method.

The number of rounds in each season for each league is plotted as a function of time. While this figure is a little convoluted, it is useful to observe potential gaps in the dataset.

<p align='center'>
  <img 
    width='600'
    src='README-images/rounds.png'
  >
</p>

It is to be expected that each league plays 30-46 rounds, so it is evident the Championship is a problematic dataset. After inspecting, it can be seen that there is incomplete data for the years 1990-1994 and 1998 (which will be dropped), and missing data between 2006 and 2020. The drop in rounds in the 2. Bundesliga in 1992 may initially be understood as incomplete data, however after futher research it is apparent that this is due to the league being briefly split in two after teams from East Germany joined the league. The majority of the data for the Eerste Divisie and Segunda Liga is missing as each season only contains information on the first round. As a result of this, little usefeul information can be able to be extracted so data from these two leagues will be dropped. The final two dips in total rounds played can be explained by a reduction in the number of matches during the COVID-19 pandemic and the data being collected while the 2021 campaign was still in progression.

This leave 127416 rows of data from 12 leagues to analyse, and brings the number of unique home teams and away teams to 499 and 498 respectively.

### Analysis

Combining the home and away goals, a bar chart can be produced to show the average number of goals per game for the various leagues. As can be seen, all the leagues have a similar average of around 2.5 goals per game, with the Dutch Eredivisie coming in at the highest with an avergae of about 3, and the French Ligue 2 the lowest at roughly 2.3 goals per game.

<p align='center'>
  <img 
    width='400'
    src='README-images/average_goals.png'
  >
</p>

By creating a new column in the dataframe and assigning each home win, draw and away win a score of 1, 0 and 1 respectively, a line graph can be plotted to see how the average outcome of all leagues has changed over the years. In theory, this value would remain at 0 indicating an equal number of home and away wins, however in reality it is shown that earlier years have a heavy bias towards home wins. What is interesting is that over the years, this bias is reducing leading to a much more equal probability of the away team winning especially in recent years.

<p align='center'>
  <img 
    width='400'
    src='README-images/outcome-over-time.png'
  >
</p>

In order to investigate the effect that the stadium size has on the outcome of a game, a scatter graph has been plotted with an ordinary least squared (OLS) regression trendline. The positive gradient of this line demonstrates that the home team is more likely to win when their stadium is larger, however this may partly be a result of stronger teams having larger capacity stadiums.

<p align='center'>
  <img 
    width='400'
    src='README-images/capacity.png'
  >
</p>

Similarly, the number of cards per game a home team receives is plotted as a function of stadium capacity. Here the trend is less pronounced, but  demonstrating a negative coefficient nonetheless. This is perhaps due to teams playing in smaller stadium being from lower leagues where the game is potentially a little more 'old school', resulting in more cardable offences. Furthermore, the home team is less likely to receive cards in comparison the away team.

<p align='center'>
  <img 
    width='400'
    src='README-images/cards.png'
  >
</p>

Other data that could be looked at is the frequency of cards given by each referee, the number of draws per league and if the time of year affects the goals/outcome.

### Hypothesis

- The feature with the greatest impact on the result will be the past results of each club. Other important features will be the past number of goals score and conceeded and the stadium size.
- The features that will affect the yellow and red cards in a game will be the past number of cards the club has received and the referee of that game.
- Stadium size will have a impact on both these labels - the larger the stadium the larger the liklelihood of a home win and less cards received. 

More features will be added and inspected as the project progresses.

## Milestone 3: Feature Engineering

### ELO

The ELO is a points system which is given to each team in relation their previous results where the stronger a team is, the greater their ELO value. This value is calculated by a predetermined third party algorithm after each game, but comparing the two teams score will almost certainly be a key feature in predicting the outcome of games. This data is loaded using ```pickle``` and merged into the main dataframe.
```python 
elo_dict = pickle.load(open('elo_dict.pkl', 'rb'))
elo_df = pd.DataFrame.from_dict(elo_dict)
elo_df = elo_df.transpose().reset_index().rename(columns={'index': 'Link'})
```

### Goals Scored So Far

As stated in the hypothesis, the number of goals scored that season by a team prior to the fixture taking place is highly likely to influence the number of goals scored by said team and hence, the outcome of the match. Calculating this is relatively simply, but to create a singular function that will iterate over the entire ```main_df``` dataframe is a little more complex and requires several nested ```for``` loops.

- Initially, one club was looked at in a dataframe from a singular season (```season_df```) in order to establish the innermost ```for``` loop.  The goals scored by this team so far was calculated by creating a list with an initial value of 0. The next value is calculated by summing the previous two values, and so the number of goals scored by the team after round 1 will correspond to the value of index 1 in the list.

    ```python
  for j in season_df['round'].unique():
    match = season_df.loc[(season_df['round'] == j) & ((season_df['home_team'] == team) | (season_df['away_team'] == team))]
    if match.home_team.item() == team:
        scored_list.append(match.home_goals.item())
        scored_list[j] += scored_list[j-1]
    ```
    > Calculating the total goals scored by one team in the competition.

- These value are then inserted into ```main_df```.
- This loop is iterated over all teams in the current season, before being iterated over all seasons in the league before finally over all leagues in the dataframe.

### Goals Conceeded So Far

Simlarly, the goals conceeded by a team in the games prior will be a good indicator of the number of goals a team is likely to conceede in an upcoming game. This is calculated almost identically to goals scored so far, simply accessed the away teams goals scored for goals conceeded by the home team, and vice versa. The two features are calculated in a single function, ```calculate_goals_sofar()```.

```python
if match.home_team.item() == team:
  conceeded_list.append(match.away_goals.item())
  conceeded_list[j] += conceeded_list[j-1]
```
> Calculating the goals conceeded by the home team.

This function takes an incredibly long time to iterate through the 140k rows (upwards of 2 hours), but it only needs to run once. This dataframe with the two new features is saved as ```main_df_goals_sofar.csv```.

### Points So Far

Accumulated oints so far is calculated over the course of the season for each team. 3 points are awarded for a win, 1 for a draw and 0 for a loss.

```python
if match.outcome.item() == 1:
    points_sofar_list.append(3)
elif match.outcome.item() == 0:
    points_sofar_list.append(1)
else:
    points_sofar_list.append(0)
points_sofar_list[j] += points_sofar_list[j-1]
```

This information is saved as ```main_df_points_sofar.csv```.

### Form

Form is calculated over the past 5 games for each team and inserted into ```main_df``` under the columns, ```home_form``` or ```away_form```. This is given as a string object, e.g. ```'WWDLW'```.

```python
form = '-----'
if match.outcome.item() == 1:
  form = form[1:] + 'W'
elif match.outcome.item() == 0:
  form = form[1:] + 'D'
else:
  form = form[1:] + 'L'
```

This is saved as ```main_df_form.csv```.

## Milestone 4: Uploading to a Database

### RDS

Creating a relational database in the cloud means the data can be accessed by more than just a local machine.

- Using Amazon's RDS, a new PostgreSQL database ```football-predictions``` is set up with public access.
- A ```.env``` file is created to store the hostname and password securely and the engine is connected to with the ```rds_connect()``` function using the packages ```sqlalchemy``` and ```psycopg2```.

```python
def rds_connect():
    DATABASE_TYPE = 'postgresql'
    DBAPI = 'psycopg2'
    HOST = os.environ['DB_HOST']
    USER = 'postgres'
    PASSWORD = os.environ['DB_PASSWORD']
    DATABASE = 'football-predictions'
    PORT = 5432
    return create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
```
> This function creates a connection with the database in the cloud.

- The cleaned dataframe is then uploaded to the AWS database with this connection.

```python
def upload_initial_data():
    engine = rds_connect()
    clean_dataset = pd.read_csv('project/dataframes/cleaned_dataset.csv', index_col=0)
    df_name = 'match-results'
    clean_dataset.to_sql(df_name, engine, if_exists='replace', index=False)
```
> The ```upload_initial_data()``` function.

### Developing the Pipeline

The function ```create_cleaned_dataset()``` will read the ```.csv``` files containing the features to be inspected and merge them into one dataframe which we can use to train a model. Using the ```.join``` method, we can merge these dataframes on index and save it as a new ```.csv``` file.

```python
cleaned_dataset = goals_sofar.join(points_sofar).join(form)
cleaned_dataset.to_csv('project/dataframes/cleaned_dataset.csv')
```
> The ```create_cleaned_dataset()``` function in ```feature_engineering.py```.

A new file ```pipeline.py``` is created in which the function ```additional_data_pipeline()``` can be run whenever new data is added. This takes in the path to additional match and ELO data, merges them with ```match_info.csv``` and ```team_info.csv``` with a left join before calculating the features to be used in the training of the model. Finally, it appends this DataFrame to the PostgreSQL table in the cloud.

```python
def additional_data_pipeline(dataset_path, elo_path):
  main_df_additional = perform_eda(dataset_path, elo_path)
  create_cleaned_dataset(main_df_additional).to_csv('project/dataframes/cleaned_dataset_additional.csv')
  upload_additional_data()
```
> The pipeline used to automatically add new data to the AWS database in the cloud.

## Milestone 5: Model Training

### Baseline Score

First, a simple model is trained to obtain an initial base score which can be improved upon later. The design matrix ```X``` contains samples represented as rows and samples represented as rows, making it of size (120581, 10). The target values ```y``` represent the discrete set of values for classification, in this case a 1D array of length 120581 with values 1, 0 or -1.

To estimate how well a model performs on unseen data, the initial dataset into two: one for training and the other for testing. This testing set is used for evaluating whether a model meets necessary requirements and estimating real world performance. A test size of 0.2 will be used to represent the proportion of the dataset to be included in the test split, giving 24117 testing values which is more than enough. ```random_state``` will be set to an artbitrary integer for reproducible results.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(accuracy_score(y_test, y_pred))
```
> Calculating the accuracy score of the baseline model.

This gives a baseline score of 0.48903263258282537.

### Feature Selection

### Training multiple methods

### Picking the best method

### Testing model on testing set

## Milestone 6: Inference
