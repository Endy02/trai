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
from trai.tools.logger import Logger


class Trai():
    """
        Prepare Regression datasets
    """
    def __init__(self, dataset, index=None):
        self.logger = Logger()
        self.df = pd.read_csv(dataset, index_col=index)

    def prepare_df(self):
        pass    