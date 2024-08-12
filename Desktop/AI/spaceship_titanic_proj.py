import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

titanic_data = pd.read_csv(r'C:\Users\lando\Downloads\spaceship-titanic\train.csv')

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.21)
for train_indices, test_indices in split.split(titanic_data, titanic_data['Transported']):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class prepare(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X['Destination'] = X['Destination'].fillna('Missing')
    X['VIP'] = X['VIP'].fillna(False)
    X['CryoSleep'] = X['CryoSleep'].fillna(False)
    X['HomePlanet'] = X['HomePlanet'].fillna('Missing')

    X[["Deck", "Cabin_num", "Side"]] = X["Cabin"].str.split("/", expand=True)

    X['Deck'] = X['Deck'].fillna('Missing')
    X['Side'] = X['Side'].fillna('Missing')





    return X



class Imputermain(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    imputer = SimpleImputer(strategy="median")
    X['Age'] = imputer.fit_transform(X[['Age']])

    X['RoomService'] = imputer.fit_transform(X[['RoomService']])

    X['ShoppingMall'] = imputer.fit_transform(X[['ShoppingMall']])
    X['Spa'] = imputer.fit_transform(X[['Spa']])
    X['VRDeck'] = imputer.fit_transform(X[['VRDeck']])
    X['FoodCourt'] = imputer.fit_transform(X[['FoodCourt']])



    return X


from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    encoder = OneHotEncoder()
    matrix = encoder.fit_transform(X[["Destination"]]).toarray()

    column_names = ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e", "noDestination"]
    for i in range(len(matrix.T)):
      X[column_names[i]] = matrix.T[i]

    matrix2 = encoder.fit_transform(X[['VIP']]).toarray()
    column_names = ["isVIP", "notVIP"]
    for i in range(len(matrix2.T)):
      X[column_names[i]] = matrix2.T[i]

    matrix3 = encoder.fit_transform(X[['CryoSleep']]).toarray()
    column_names = ["isCryoSleep", "notCryoSleep"]
    for i in range(len(matrix3.T)):
      X[column_names[i]] = matrix3.T[i]

    matrix4 = encoder.fit_transform(X[['HomePlanet']]).toarray()
    column_names = ["Mars", "Earth", "Europa", "noHome"]
    for i in range(len(matrix4.T)):
      X[column_names[i]] = matrix4.T[i]

    matrix5 = encoder.fit_transform(X[['Side']]).toarray()
    column_names = ["Port", "Star", "noside"]
    for i in range(len(matrix5.T)):
      X[column_names[i]] = matrix5.T[i]

    matrix6 = encoder.fit_transform(X[['Deck']]).toarray()
    column_names = ["B", "F", "A", "G", "nodeck", "E", "D", "C", "T"]
    for i in range(len(matrix6.T)):
      X[column_names[i]] = matrix6.T[i]

    

    return X


class FeatureDropper(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    return X.drop([ 'PassengerId', 'noside', 'Side', 'Deck', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination',"VIP","Name","Cabin_num"], axis=1, errors='ignore')


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
  ('prepare', prepare()),
  ('ageimputer', Imputermain()),
  ('featureencoder', FeatureEncoder()),
  ('featuredropper', FeatureDropper())
])

strat_train_set = pipeline.fit_transform(strat_train_set)
strat_test_set = pipeline.fit_transform(strat_test_set)

strat_test_set.info()

titanic_data_testing = pd.read_csv(r'C:\Users\lando\Downloads\spaceship-titanic\test.csv')
titanic_test_final = pipeline.fit_transform(titanic_data_testing)
titanic_test_final.info()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_data_final_testing = scaler.fit_transform(titanic_test_final)


from sklearn.preprocessing import StandardScaler

X = strat_train_set.drop("Transported", axis=1)
y = strat_train_set["Transported"]

X.info()
y.info()

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = y.to_numpy()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

prod_clf = RandomForestClassifier()

param_grid = [
    {"n_estimators": [1000, 1200], "max_depth": [15,20], "min_samples_split": [4,5]}
]

grid_search = GridSearchCV(prod_clf, param_grid, cv=3,
                           scoring="accuracy",
                           return_train_score=True)

grid_search.fit(X_data, y_data)

final_clf = grid_search.best_estimator_


predictions = final_clf.predict(X_data_final_testing)

final_df = pd.DataFrame(titanic_data_testing['PassengerId'])
final_df['Transported'] = predictions
final_df.to_csv((r'C:\Users\lando\Downloads\spaceship-titanic\submissionIdle.csv'), index=False)


