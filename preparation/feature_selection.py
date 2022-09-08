import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cleaned_dataset = pd.read_csv('preparation/dataframes/cleaned_dataset.csv', index_col=0)
X = cleaned_dataset.drop('outcome', axis=1)
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

    home_win_bar = px.bar(
        x=X.columns,
        y=home_win_coefficients,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='Home Win Logistic Regresssion Coefficients')
    home_win_bar.update_layout(template='plotly_dark')
    home_win_bar.write_image('README-images/home_win_coefficients.png')

    draw_bar = px.bar(
        x=X.columns,
        y=draw_coefficients,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='Draw Logistic Regresssion Coefficients')
    draw_bar.update_layout(template='plotly_dark')
    draw_bar.write_image('README-images/draw_coefficients.png')

    away_win_bar = px.bar(
        x=X.columns,
        y=away_win_coefficients,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='Away Win Logistic Regresssion Coefficients')
    away_win_bar.update_layout(template='plotly_dark')
    away_win_bar.write_image('README-images/away_win_coefficients.png')


def plot_importance():
    """Plot the importance of each feature using the DecisionTreeRegressor model.
    Plots saved to README-images.
    """
    model = DecisionTreeClassifier(random_state=13)
    model.fit(X, y)
    importance = model.feature_importances_

    importance_bar = px.bar(
        x=X.columns,
        y=importance,
        labels={'x': 'feature', 'y': 'coefficient'},
        title='The importance score of each feature using a DecisionTreeClassifier model')
    importance_bar.update_layout(template='plotly_dark')
    importance_bar.write_image('README-images/importance_coefficients.png')


def test_new_model(X):
    """Fits a new model with a different combination of features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def plot_feature_combinations():
    none_removed_accuracy = test_new_model(X)
    no_elo_accuracy = test_new_model(X=X.drop(['elo_home', 'elo_away'], axis=1))
    no_form_accuracy = test_new_model(X=X.drop(['home_form', 'away_form'], axis=1))
    no_points_accuracy = test_new_model(X=X.drop(['home_points_sofar', 'away_points_sofar'], axis=1))
    no_ave_scored_accuracy = test_new_model(X=X.drop(['average_recent_home_scored', 'average_recent_away_scored'], axis=1))
    no_ave_conceeded_accuracy = test_new_model(X=X.drop(['average_recent_home_conceeded', 'average_recent_away_conceeded'], axis=1))
    fig = px.bar(
        x=[none_removed_accuracy, no_elo_accuracy, no_form_accuracy, no_points_accuracy, no_ave_scored_accuracy, no_ave_conceeded_accuracy],
        y=['elo', 'none', 'form', 'points', 'ave_scored', 'ave_conceeded'],
        labels={'x': 'accuracy score', 'y': 'features removed'},
        title='The accuracy score of logistic regression with various features removed')
    fig.update_layout(template='plotly_dark')
    fig.update_xaxes(range=[0.48, 0.5])
    fig.write_image('README-images/feature-removal.png')

if __name__ == '__main__':
    # plot_coefficients()
    plot_feature_combinations()
    # plot_importance()