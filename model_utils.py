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

def switch_case(argument):
    switcher = {
        "Random Forest Classifier": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Adaboost Classifier": AdaBoostClassifier(),
        "Extra Trees Classifier": ExtraTreesClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
        "Gaussian Naive Bayes Classifier": GaussianNB(),
        "Bernoulli Naive Bayes Classifier": BernoulliNB(),
        "Multinomial Naive Bayes Classifier": MultinomialNB(),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(),
        "Bagging Classifier": BaggingClassifier(),
        "XGBoost Classifier": XGBClassifier(),
        "LightGBM Classifier": LGBMClassifier(),
        "CatBoost Classifier": CatBoostClassifier(),
        "MLP Classifier": MLPClassifier()
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

def run_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    st.write(f"**{model_name} Performance Metrics:**")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    st.write(f"Log Loss: {log_loss(y_test, y_score) if y_score is not None else 'N/A'}")

    cm(y_test, y_pred)
    if y_score is not None:
        aoc(y_score, len(np.unique(y_test)), y_test)

def cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

def aoc(y_score, n_classes, y_test):
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='best')
    st.pyplot(fig)
