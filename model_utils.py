# model_utils.py

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, 
    RandomForestClassifier, ExtraTreesClassifier, IsolationForest
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet, Lasso, LogisticRegression, PassiveAggressiveClassifier, 
    Perceptron, RidgeClassifier, SGDClassifier
)
from sklearn.metrics import (
    accuracy_score, auc, cohen_kappa_score, confusion_matrix, f1_score, 
    log_loss, matthews_corrcoef, precision_score, recall_score, 
    roc_auc_score, roc_curve
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

def switch_case(argument, params):
    switcher = {
        "Random Forest Classifier": RandomForestClassifier(**params),
        "SVM": SVC(**params),
        "Decision Tree Classifier": DecisionTreeClassifier(**params),
        "Logistic Regression": LogisticRegression(**params),
        "Adaboost Classifier": AdaBoostClassifier(**params),
        "Extra Trees Classifier": ExtraTreesClassifier(**params),
        "Gradient Boosting Classifier": GradientBoostingClassifier(**params),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(**params),
        "Gaussian Naive Bayes Classifier": GaussianNB(),
        "Bernoulli Naive Bayes Classifier": BernoulliNB(),
        "Multinomial Naive Bayes Classifier": MultinomialNB(),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(**params),
        "Bagging Classifier": BaggingClassifier(**params),
        "XGBoost Classifier": XGBClassifier(**params),
        "LightGBM Classifier": LGBMClassifier(**params),
        "CatBoost Classifier": CatBoostClassifier(**params),
        "MLP Classifier": MLPClassifier(**params)
    }
    return switcher.get(argument)

def pre_processing(df, columns):
    encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = encoder.fit_transform(df[column])

    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)
    return pd.DataFrame(df, columns=columns)

def run_model(df, model, model_name):
    # Assuming the code for run_model is similar to the one provided earlier
    pass

def cm(y_true, y_pred):
    # Function to generate confusion matrix
    pass

def aoc(y_score, n_classes, y_test):
    # Function to generate ROC curve
    pass
