from sklearn import metrics  # for evaluation
# from sklearn import svm  # for Discriminator
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer  # To replace null value in dataframe
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # for fine-tuning
# from sklearn.preprocessing import StandardScaler  # for feature scaling
import pandas as pd
import numpy as np
import os, json

class MissingVal():
    def __init__(self):
        self.data_url = self.train_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/missing_val/"
        self.test_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/missing_val/test.csv"
        self.train_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/missing_val/train.csv"

    def process(self):
        # Read the data
        X_full = pd.read_csv(self.train_data, index_col='Id')
        X_test_full = pd.read_csv(self.test_data, index_col='Id')

        # Remove rows with missing target, separate target from predictors
        X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
        y = X_full.SalePrice
        X_full.drop(['SalePrice'], axis=1, inplace=True)

        # To keep things simple, we'll use only numerical predictors
        X = X_full.select_dtypes(exclude=['object'])
        X_test = X_test_full.select_dtypes(exclude=['object'])

        # Break off validation set from training data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                            random_state=0)
        print(' ---- Before treatment ----')
        # Shape of training data (num_rows, num_columns)
        print(f"Training set shape : {X_train.shape}")
        print(f"Training set columns : {X_train.shape[1]}")
        # Number of missing values in each column of training data
        missing_val_count_by_column = (X_train.isnull().sum())
        print(f"Total mission values : {X_train.isnull().sum().sum()}")
        print(f"Total of missing value columns : {len(missing_val_count_by_column[missing_val_count_by_column > 0])}")
        print(missing_val_count_by_column[missing_val_count_by_column > 0])
        print(' ---- END Before treatment ----')

        miss_val = missing_val_count_by_column[missing_val_count_by_column > 0].index
        print(missing_val_count_by_column[missing_val_count_by_column > 0].index)

        reduced_X_train = X_train.drop(miss_val, axis=1)
        reduced_X_valid = X_valid.drop(miss_val, axis=1)

        print(f"Reduce X train : {reduced_X_train.shape} | Reduce X Valid {reduced_X_valid.shape}")
        print("MAE (Drop columns with missing values):")
        print(self.score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_X_train = imputer.fit_transform(X_train)
        imputed_X_valid = imputer.transform(X_valid)

        print(' ---- Simple Imputer ----')
        # Fill in the lines below: imputation removed column names; put them back
        df_X_train = pd.DataFrame(data=imputed_X_train, columns=X_train.columns)
        df_X_valid = pd.DataFrame(data=imputed_X_valid, columns=X_valid.columns)
        
        print("MAE (Imputation):")
        print(self.score_dataset(df_X_train, df_X_valid, y_train, y_valid))
    
        print(' ---- Simple Imputer ----')

    # Function for comparing different approaches
    def score_dataset(self, X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return metrics.mean_absolute_error(y_valid, preds)