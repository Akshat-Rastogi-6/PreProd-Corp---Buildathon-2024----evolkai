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
        "Adaboost Classifier" : AdaBoostClassifier(),
        "Extra Trees Classifier" : ExtraTreeClassifier(),
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

def cm(y_test, y_pred):
    mdl_cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(13, 12))
    sns.heatmap(mdl_cm, annot=True)
    plt.savefig('uploads/confusion_matrix.jpg', format="jpg", dpi=300)
    st.image("uploads/confusion_matrix.jpg", caption="Confusion Matrix of your Data", width=600)

def pre_processing(df, columns):
    encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = encoder.fit_transform(df[column])

    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)
    return pd.DataFrame(df, columns=columns)

def run_model(df, model, model_name):
    st.subheader(model_name)
    target_column = st.text_input("Enter your target column : ")
    
    if target_column != "":
        X = df.drop(columns=[target_column])
        y = df[target_column]
        testing_size = st.text_input("Enter the test splitting size : ")
        if testing_size != "":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testing_size), random_state=42)
            model.fit(X_train, y_train)
            testing_file = st.sidebar.file_uploader("Upload your testing CSV file...", type=['csv'])
            
            if testing_file is not None:
                test = pd.read_csv(testing_file)
                test = pre_processing(test, test.columns)
                y_pred = model.predict(test)
            
            y_pred = model.predict(X_test)
            eval_mat = st.sidebar.selectbox("Select your Evaluation Matrix :", [
                "Accuracy", "Precision", "Recall(Sensitivity)", "F1 Score", 
                "Roc AUC Score", "Cohen's Kappa", "Matthew's Correlation Coefficient", 
                "Log Loss"
            ])

            metrics = {
                "Accuracy": lambda: accuracy_score(y_test, y_pred),
                "Precision": lambda: precision_score(y_test, y_pred, average='weighted'),
                "Recall(Sensitivity)": lambda: recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": lambda: f1_score(y_test, y_pred, average='weighted'),
                "Roc AUC Score": lambda: roc_auc_score(y_test, y_pred),
                "Cohen's Kappa": lambda: cohen_kappa_score(y_test, y_pred),
                "Matthew's Correlation Coefficient": lambda: matthews_corrcoef(y_test, y_pred),
                "Log Loss": lambda: log_loss(y_test, y_pred)
            }

            if eval_mat in metrics:
                result = metrics[eval_mat]()
                st.write(f"{eval_mat} :", float(result))

            if eval_mat:
                curves = st.sidebar.selectbox("Select the metrics you want to see : ", ["Confusion Matrix", "ROC Curve"])
                if curves == "Confusion Matrix":
                    cm(y_test, y_pred)
                if curves == "ROC Curve":
                    y_arr = np.array(y)
                    unique_classes, counts = np.unique(y_arr, return_counts=True)
                    n_classes = len(unique_classes)
                    y_train_bin = label_binarize(y_train, classes=range(n_classes))
                    classifier = OneVsRestClassifier(model)
                    classifier.fit(X_train, y_train_bin)
                    y_score = classifier.predict_proba(X_test)
                    aoc(y_score, n_classes, y_test)

def aoc(y_score, n_classes, y_test):
    y_test_np = y_test.to_numpy()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_np == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiclass Data')
    plt.legend(loc="lower right")
    plt.savefig('uploads/roc.jpg', format="jpg", dpi=300)
    st.image("uploads/roc.jpg", caption="ROC of your Data", width=600)
