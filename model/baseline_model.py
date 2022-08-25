import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cleaned_dataset = pd.read_csv('preparation/dataframes/cleaned_dataset.csv')
X = cleaned_dataset[[
    'round', 'home_scored_sofar', 'away_scored_sofar', 'home_conceeded_sofar', 'away_conceeded_sofar',
    'home_points_sofar', 'away_points_sofar', 'elo_home', 'elo_away', 'home_form', 'away_form']]
y = cleaned_dataset['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
dump(model, 'model/baseline.joblib') 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))