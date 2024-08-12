import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Assume the file you uploaded is named 'test.csv'
titanic_data = pd.read_csv(r'C:\Users\lando\Downloads\titanic\train.csv')

titanic_data.describe()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data["Survived"]):
  strat_train_set = titanic_data.loc[train_indices]
  strat_test_set = titanic_data.loc[test_indices]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    imputer = SimpleImputer(strategy="mean")
    X['Age'] = imputer.fit_transform(X[['Age']])
    return X


from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    encoder = OneHotEncoder()
    matrix = encoder.fit_transform(X[["Embarked"]]).toarray()

    column_names = ["C", "S", "Q", "N"]
    for i in range(len(matrix.T)):
      X[column_names[i]] = matrix.T[i]

    matrix = encoder.fit_transform(X[['Sex']]).toarray()
    column_names = ["Male", "Female"]
    for i in range(len(matrix.T)):
      X[column_names[i]] = matrix.T[i]

    return X


class FeatureDropper(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    return X.drop([ 'Name', 'Ticket', 'Cabin', 'Embarked', 'Sex',"N"], axis=1, errors='ignore')




from sklearn.pipeline import Pipeline

pipeline = Pipeline([
  ('ageimputer', AgeImputer()),
  ('featureencoder', FeatureEncoder()),
  ('featuredropper', FeatureDropper())
])

strat_train_set = pipeline.fit_transform(titanic_data)

from sklearn.preprocessing import StandardScaler

X = strat_train_set.drop("Survived", axis=1)
y = strat_train_set["Survived"]

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = y.to_numpy()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

prod_clf = RandomForestClassifier()

param_grid = [
    {"n_estimators": [10,100,200,500], "max_depth": [None,5, 10], "min_samples_split": [2,3,4]}
]

grid_search = GridSearchCV(prod_clf, param_grid, cv=3,
                           scoring="accuracy",
                           return_train_score=True)

grid_search.fit(X_data, y_data)

final_clf = grid_search.best_estimator_

strat_test_set = pipeline.fit_transform(strat_test_set)

X_test = strat_test_set.drop(["Survived"], axis=1)
y_test = strat_test_set["Survived"]

scaler = StandardScaler()
X_test_data = scaler.fit_transform(X_test)
y_test_data = y_test.to_numpy()

print(final_clf.score(X_test_data, y_test_data))

