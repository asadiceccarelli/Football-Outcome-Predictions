import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cleaned_dataset = pd.read_csv('preparation/dataframes/cleaned_dataset.csv')
X = cleaned_dataset[[
    'round', 'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'home_points_sofar', 'away_points_sofar', 'elo_home', 'elo_away', 'home_form', 'away_form']]
y = cleaned_dataset['outcome']


def plot_coefficients():
    """Plot the logistic regression coefficients for each outcome.
    Plots saved to README-images.
    """
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(X, y)
    home_win_coefficients = model.coef_[0]
    draw_coefficients = model.coef_[1]
    away_win_coefficients = model.coef_[2]

    px.bar(
        x=X.columns,
        y=home_win_coefficients,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='Home Win Logistic Regresssion Coefficients').write_image('README-images/home_win_coefficients.png')

    px.bar(
        x=X.columns,
        y=draw_coefficients,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='Draw Logistic Regresssion Coefficients').write_image('README-images/draw_coefficients.png')

    px.bar(
        x=X.columns,
        y=away_win_coefficients,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='Away Win Logistic Regresssion Coefficients').write_image('README-images/away_win_coefficients.png')


def plot_importance():
    """Plot the importance of each feature using the DecisionTreeRegressor model.
    Plots saved to README-images.
    """
    model = DecisionTreeClassifier(random_state=13)
    model.fit(X, y)
    importance = model.feature_importances_

    px.bar(
        x=X.columns,
        y=importance,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='The importance score of each feature using a DecisionTreeClassifier model').write_image('README-images/importance_coefficients.png')


def test_new_model(X):
    """Fits a new model with a different combination of features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def plot_feature_combinations():
    none_removed_accuracy = test_new_model(X[[
    'round', 'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'home_points_sofar', 'away_points_sofar', 'elo_home', 'elo_away', 'home_form', 'away_form']])
    print(none_removed_accuracy)
    no_round_accuracy = test_new_model(X[[
    'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'home_points_sofar', 'away_points_sofar', 'elo_home', 'elo_away', 'home_form', 'away_form']])
    no_form_accuracy = test_new_model(X[[
    'round', 'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'home_points_sofar', 'away_points_sofar', 'elo_home', 'elo_away']])
    no_points_accuracy = test_new_model(X[[
    'round', 'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'elo_home', 'elo_away', 'home_form', 'away_form']])
    print(no_points_accuracy)
    no_round_form_accuracy = test_new_model(X[[
    'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'home_points_sofar', 'away_points_sofar', 'elo_home', 'elo_away']])
    no_round_points_accuracy = test_new_model(X[[
    'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'elo_home', 'elo_away', 'home_form', 'away_form']])
    no_form_points_accuracy = test_new_model(X[[
    'round', 'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'elo_home', 'elo_away']])
    no_round_form_points_accuracy = test_new_model(X[[
    'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'elo_home', 'elo_away']])
    fig = px.bar(
        x=[
            none_removed_accuracy, no_round_accuracy, no_form_accuracy, no_points_accuracy, no_round_form_accuracy, 
            no_round_points_accuracy, no_form_points_accuracy, no_round_form_points_accuracy],
        y=['none', 'round', 'form', 'points', 'round + form', 'round + points', 'form + points', 'round + form + points'],
        orientation='h',
        labels={'x': 'accuracy score', 'y': 'features removed'},
        title='The accuracy score of logistic regression with various features removed.')
    fig.update(layout_xaxis_range = [0.492, 0.4945])
    fig.write_image('README-images/feature-removal.png')

if __name__ == '__main__':
    plot_feature_combinations()