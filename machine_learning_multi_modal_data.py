import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from util import dict_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
class Model():  
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=400, random_state=42, min_samples_leaf=7, min_samples_split=10, class_weight="balanced")
        
    def fit(self, X_dict, y):
        
        tabular = X_dict['tabular']
        
        #Fill NaNs with mode of each column
        tabular.fillna(tabular.mode().iloc[0], inplace=True)
        
        def remove_c(x):
            if isinstance(x, str):
                return int(x.replace('C', ''))
            elif isinstance(x, (int, float)):
                return x
            else:
                raise TypeError(f"Unsupported type: {type(x)}")
        # Apply the remove_c function to columns with object dtype
        non_numeric_cols = tabular.select_dtypes(include=['object']).columns.tolist()
        tabular[non_numeric_cols] = tabular[non_numeric_cols].applymap(remove_c)
        
        # Removing highly correlated features
        corr_matrix = tabular.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        tabular = tabular.drop(columns=to_drop)
        
        # Scaling
        scaler = StandardScaler()
        tabular = pd.DataFrame(scaler.fit_transform(tabular))
        
        images = X_dict['images'].copy()
        images = np.nan_to_num(images)
        
        flattened_images = images.reshape(images.shape[0], -1)
        for i in range(flattened_images.shape[1]):
            tabular[f'img_{i}'] = flattened_images[:, i]
        tabular['target'] = y
        ones = tabular[tabular['target'] == 1]
        zeros = tabular[tabular['target'] == 0]
        ones = ones.sample(len(tabular[tabular['target'] == 0]), replace = True,  random_state=1)
        tabular =  pd.concat([zeros, ones], axis = 0)
        corr_matrix = tabular.corr()


        target_corr = corr_matrix['target']
        target_corr_sorted = target_corr.sort_values(ascending=False)
        
        na_mask = target_corr_sorted.isna()
        no_correlation = na_mask[na_mask == True].index.tolist()
        
        self.columns_to_remove = no_correlation
        tabular = tabular.drop(columns=self.columns_to_remove)
        
        self.model.fit(tabular.iloc[:, :-1], tabular['target'])
        return self.model
        
    def predict(self, X_dict):
        
        tabular = X_dict['tabular']
        
        #Fill NaNs with mode of each column
        tabular.fillna(tabular.mode().iloc[0], inplace=True)
        
        def remove_c(x):
            if isinstance(x, str):
                return int(x.replace('C', ''))
            elif isinstance(x, (int, float)):
                return x
            else:
                raise TypeError(f"Unsupported type: {type(x)}")
        # Apply the remove_c function to columns with object dtype
        non_numeric_cols = tabular.select_dtypes(include=['object']).columns.tolist()
        tabular[non_numeric_cols] = tabular[non_numeric_cols].applymap(remove_c)
        
        # Removing highly correlated features
        corr_matrix = tabular.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        tabular = tabular.drop(columns=to_drop)
        
        scaler = StandardScaler()
        tabular = pd.DataFrame(scaler.fit_transform(tabular))
        
        images = X_dict['images'].copy()
        images = np.nan_to_num(images)
        
        flattened_images = images.reshape(images.shape[0], -1)
        for i in range(flattened_images.shape[1]):
            tabular[f'img_{i}'] = flattened_images[:, i]
        tabular = tabular.drop(columns=self.columns_to_remove)
        return self.model.predict(tabular)