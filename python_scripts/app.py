import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.calibration import label_binarize
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,RandomForestClassifier)
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Get the selected model, file path, and target variable name from the command line arguments
selected_model = sys.argv[1]
file_path = sys.argv[2]
target_variable = sys.argv[3]
drop = sys.argv[4]

# selected_model = '36_br' #sys.argv[1]
# # # file_path = r"C:\Artificial intelligence\Machine learning\heart\heart.csv"#sys.argv[2]
# file_path = r"D:\datasets\polymer.csv"
# target_variable = 'log(viscosity) in cP'#sys.argv[3]
# # drop = 'discoveryDate'
