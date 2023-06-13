from sklearn import metrics  # for evaluation
# from sklearn import svm  # for Discriminator
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer  # To replace null value in dataframe
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # for fine-tuning
from sklearn.preprocessing import OrdinalEncoder, StandardScaler  # for feature scaling
import pandas as pd
import numpy as np
import os, json

class Categoricalvar():
    def __init__(self):
        self.data_url = self.train_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/missing_val/"
        self.test_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/missing_val/test.csv"
        self.train_data = os.path.abspath(os.path.dirname('data'))+ "/data/exercice/missing_val/train.csv"

    def process(self):
        # Read the data
        X = pd.read_csv(self.train_data, index_col='Id')
        X_test = pd.read_csv(self.test_data, index_col='Id')

        # Remove rows with missing target, separate target from predictors
        X.dropna(axis=0, subset=['SalePrice'], inplace=True)
        y = X.SalePrice
        X.drop(['SalePrice'], axis=1, inplace=True)

        # To keep things simple, we'll drop columns with missing values
        cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
        X.drop(cols_with_missing, axis=1, inplace=True)
        X_test.drop(cols_with_missing, axis=1, inplace=True)

        # Break off validation set from training data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                            train_size=0.8, test_size=0.2,
                                                            random_state=0)
        print(f"Shape before droping | X_train : {X_train.shape} , X_valid : {X_valid.shape}")
        drop_X_train = X_train.drop(X_train.select_dtypes(include=['object']), axis=1)
        drop_X_valid = X_valid.drop(X_valid.select_dtypes(include=['object']), axis=1)
        
        print(' ---- Droping categorical columns ----')
        print(drop_X_train.shape)
        print(drop_X_valid.shape)
        print(' ---- Droping categorical columns ----\n')

        print("MAE from Approach 1 (Drop categorical variables):")
        print(self.score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
        print("\n")

        print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
        print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

        # Categorical columns in the training data
        object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

        # Columns that can be safely ordinal encoded
        good_label_cols = [col for col in object_cols if 
                        set(X_valid[col]).issubset(set(X_train[col]))]
                
        # Problematic columns that will be dropped from the dataset
        bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
        print('Categorical columns that will be ordinal encoded:', good_label_cols)
        print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

        # Drop categorical columns that will not be encoded
        label_X_train = X_train.drop(bad_label_cols, axis=1)
        label_X_valid = X_valid.drop(bad_label_cols, axis=1)

        print(f"X_train : {X_train.shape} | X_valid : {X_valid.shape}")
        print("\n")

        encoder = OrdinalEncoder()
        label_X_train[good_label_cols] = encoder.fit_transform(X_train[good_label_cols]) 
        label_X_valid[good_label_cols] = encoder.transform(X_valid[good_label_cols]) 

        print("MAE from Approach 2 (Ordinal Encoding):") 
        print(self.score_dataset(label_X_train, label_X_valid, y_train, y_valid))
        print("\n")

        # Get number of unique entries in each column with categorical data
        object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
        d = dict(zip(object_cols, object_nunique))

        # Print number of unique entries by column, in ascending order
        print(sorted(d.items(), key=lambda x: x[1]))
        


    # function for comparing different approaches
    def score_dataset(self, X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return metrics.mean_absolute_error(y_valid, preds)