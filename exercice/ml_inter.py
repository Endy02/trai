import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for coloring
from sklearn import metrics  # for evaluation
# from sklearn import svm  # for Discriminator
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFE
# from sklearn.impute import SimpleImputer  # To replace null value in dataframe
from sklearn.linear_model import SGDRegressor, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, TheilSenRegressor, LinearRegression, LassoLars, HuberRegressor, Ridge   # For data analizis
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split  # for fine-tuning
# from sklearn.preprocessing import StandardScaler  # for feature scaling
import pandas as pd
import numpy as np
import os, json



class MlInter():
    def __init__(self):
      self.data_url = self.train_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/ml_inter/"
      self.test_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/ml_inter/test.csv"
      self.train_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/ml_inter/train.csv"

    def process(self):
        # Read the data
        X_full = pd.read_csv(self.train_data, index_col='Id')
        X_test_full = pd.read_csv(self.test_data, index_col='Id')
        # Obtain target and predictors
        y = X_full.SalePrice
        features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
        X = X_full[features].copy()
        X_test = X_test_full[features].copy()

        # Break off validation set from training data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

        random_scores, choice_scores = self.__model_selection(X_train, y_train, X_valid, y_valid)
        print(' ---- Random Forest model scoring ----')
        print(random_scores)
        print(' ---- END Random Forest model scoring ---- \n')

        print(' ---- Choices model scoring ----')
        print(choice_scores)
        print(' ---- END Choicesmodel scoring ---- \n')

        best_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)

        # Fit the model to the training data
        best_model.fit(X, y)

        # Generate test predictions
        preds_test = best_model.predict(X_test)

        # Save predictions in format used for competition scoring
        output = pd.DataFrame({'Id': X_test.index,
                            'SalePrice': preds_test})
        output.to_csv(self.data_url + 'submission.csv', index=False)

    def __model_selection(self, X_train, y_train, X_valid, y_valid):
        models = [RandomForestRegressor(), Ridge(), HuberRegressor(max_iter=2000), BayesianRidge(), SGDRegressor(), LassoLars(), ARDRegression(), PassiveAggressiveRegressor(), TheilSenRegressor(), LinearRegression()]
        random_model = [RandomForestRegressor(n_estimators=50, random_state=0), RandomForestRegressor(n_estimators=100, random_state=0), RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0), RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0), RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)]
        
        Y_scores = []
        rand_scores = []

        for i, m in enumerate(models):
            df_fit = m.fit(X_train, y_train)
            y_pred = df_fit.predict(X_valid)
            Y_scores.append({"id": i, "type": type(m), "score": round(metrics.mean_absolute_error(y_pred, y_valid))})

        for i, m in enumerate(random_model):
            df_fit = m.fit(X_train, y_train)
            y_pred = df_fit.predict(X_valid)
            rand_scores.append({"id": i, "type": type(m), "score": round(metrics.mean_absolute_error(y_pred, y_valid))})

        return rand_scores, Y_scores