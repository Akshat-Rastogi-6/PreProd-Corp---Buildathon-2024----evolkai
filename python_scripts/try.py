import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
from itertools import cycle

import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
# from reportlab.pdfgen import canvas
from sklearn.calibration import label_binarize
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression,PassiveAggressiveClassifier, RidgeClassifier,RidgeClassifierCV, SGDClassifier)
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, roc_curve)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import (BernoulliNB, ComplementNB, GaussianNB,MultinomialNB)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
# done

def aoc(x):
    global y
    Y = label_binarize(y, classes=[1, 2, 3, 4,  5, 6, 7, 8, 9])
    n_classes = 2
    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i],tpr[i], _ = roc_curve(Y_test, x[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test, x[:, i], pos_label=i)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    fpr["macro"], tpr["macro"], _ = roc_curve(Y_test, x[:, i], pos_label=i)
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of ROC to multiclass")
    plt.legend(loc="lower right")
    plt.savefig("uploads/aoc_plot.jpg", format="jpg", dpi=300)
    plt.close()

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

# Load the preprocessed dataset from the CSV file
data = pd.read_csv(file_path)
if drop == "NaN":
    dat = data
else:
    dat = data.drop(columns=[drop])

#data pre-processing

#label encoder
df = pd.DataFrame(dat)
# Initialize LabelEncoder
encoder = LabelEncoder()
# Iterate through each column in the dataframe
for column in df.columns:
    # Check if the column contains string values
    if df[column].dtype == 'object':
        # Fit label encoder and transform the column
        df[column] = encoder.fit_transform(df[column])


# Replace null values with column mean
converted_data = df.fillna(df.mean())

# Split the dataset into features (X) and target variable (y)
X = converted_data.drop([target_variable] , axis=1)
y = converted_data[target_variable]

# Train the model
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# Create the selected machine learning model

match selected_model:
    case '1_svm':
        model = SVC(probability= True)
        param_name = 'kernel'
    case '2_lr':
        model = LogisticRegression()
        param_name = 'max_iter'
    case '3_rf':
        model = RandomForestClassifier()
        param_name = 'n_estimators'
    case '4_abc':
        model = AdaBoostClassifier()
        param_name = 'n_estimators'
    case '5_dt':
        model = DecisionTreeClassifier()
        param_name = 'max_depth'
    case '6_etc':
        model = ExtraTreeClassifier()
        param_name = 'max_depth'
    case '7_gnb':
        model = GradientBoostingClassifier()
        param_name = 'n_estimators'
    case '8_bnb':
        model = BernoulliNB()
        param_name = 'alpha'
    case '9_mnb':
        model = MultinomialNB()
        param_name = 'alpha'
    case '10_cnb':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = ComplementNB()
        param_name = 'alpha'
    case '11_gbc':
        model = GradientBoostingClassifier()
        param_name = 'n_estimators'
    case '12_hgbc':
        model = HistGradientBoostingClassifier()
        param_name = 'max_iter'
    case  '13_knn':
        model = KNeighborsClassifier()
        param_name = 'n_neighbors'
    case '14_mlp':
        model = MLPClassifier()
        param_name = 'max_iter'
    case '15_qda':
        model = QuadraticDiscriminantAnalysis()
        param_name = 'tol'
    case '16_lda':
        model = LinearDiscriminantAnalysis()
        param_name = 'n_components'
    case '17_pac':
        model = PassiveAggressiveClassifier()
        param_name = 'tol'
    case '18_rc':
        model = RidgeClassifier()
        param_name = 'alpha'
    case '19_rcvc':
        model = RidgeClassifierCV()
        param_name = 'alphas'
    case '20_gm':
        model = GaussianMixture()
        param_name = 'n_components'
    case '21_nc':
        model = NearestCentroid()
        param_name = 'metric'
    case '22_sgdc':
        model = SGDClassifier()
        param_name = 'max_iter'
    case '23_bc':
        model = BaggingClassifier()
        param_name = 'n_estimators'
    case '24_lgbm':
        model = LGBMClassifier()
        param_name = 'n_estimators'
    case '25_xgboost':
        model = XGBClassifier()
        param_name = 'max_depth'
    case '26_l_reg':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    case '27_ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    case '28_lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)  # Alpha controls regularization strength
    case '29_elastic':
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=0.1, l1_ratio=0.5) 
    case '30_polfeat':
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(poly.transform(X_test))
    case '31_svr':
        from sklearn.svm import SVR
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    case '32_dtr':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=5)
    case '33_rfr':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
    case '34_gbr':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    case '35_xgbr':
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    case '36_br':
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge()
    case '37_kr':
        from sklearn.kernel_ridge import KernelRidge
        model = KernelRidge(alpha=1.0, kernel='linear')

        
try:
    model.fit(X_train, Y_train)

    # Make predictions on the dataset
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(Y_test, y_pred)*100
    clf = classification_report(Y_test, y_pred)

    model_1=OneVsRestClassifier(model)
    model_1.fit(X_train, Y_train)
    model_1pred_prob= model_1.predict_proba(X_test)

    def cm():
        mdl_cm=confusion_matrix(Y_test, y_pred)
        plt.figure(figsize = (13,12))
        sns.heatmap(mdl_cm, annot=True)
        plt.savefig('uploads/confusion_matrix.jpg', format="jpg", dpi=300)

    #Validation Curve

    def validation():

        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.model_selection import validation_curve

        param_range = [1, 2, 3, 4, 5]  # Range of hyperparameter values to test

        train_scores, test_scores = validation_curve(
            model,  # Your classifier here
            X_train,
            Y_train,
            param_name=param_name,
            param_range=param_range,
            cv=5,  # Number of cross-validation folds
            scoring="accuracy",  # Scoring metric
            n_jobs=-1,  # Use all available CPU cores
        )

        # Calculate mean and standard deviation for training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate mean and standard deviation for test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot validation curve
        plt.figure(figsize=(8, 6))
        plt.plot(
            param_range,
            train_mean,
            label="Training score",
            color="blue",
            marker="o",
        )
        plt.fill_between(
            param_range,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="blue",
        )
        plt.plot(
            param_range,
            test_mean,
            label="Cross-validation score",
            color="red",
            marker="o",
        )
        plt.fill_between(
            param_range,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.2,
            color="red",
        )

        plt.title("Validation Curve")
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig("uploads/valid.jpg", format="jpg", dpi=300)
        plt.close()

    # Calculate specificity, sensitivity, TPR, and FPR
    class_labels = np.unique(Y_test)
    n_classes = len(class_labels)
    specificity = {}
    sensitivity = {}
    tpr = {}
    fpr = {}

    for label in class_labels:
        tn = np.sum(np.logical_and(Y_test != label, y_pred != label))
        fp = np.sum(np.logical_and(Y_test != label, y_pred == label))
        fn = np.sum(np.logical_and(Y_test == label, y_pred != label))
        tp = np.sum(np.logical_and(Y_test == label, y_pred == label))
        
        specificity[label] = tn / (tn + fp)
        sensitivity[label] = tp / (tp + fn)
        tpr[label] = sensitivity[label]
        fpr[label] = fp / (fp + tn)
        
    # Calculate PLR and NLR for each class
    PLR = {}
    NLR = {}

    for label in class_labels:
        if specificity[label] != 0:
            PLR[label] = sensitivity[label] / (1 - specificity[label])
            NLR[label] = (1 - sensitivity[label]) / specificity[label]

    # Calculate overall PLR and NLR
    overall_PLR = np.mean(list(PLR.values()))
    overall_NLR = np.mean(list(NLR.values()))




    overall_PLR = np.mean(list(PLR.values()))
    overall_NLR = np.mean(list(NLR.values()))
    print(clf)
    mre = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100  # Convert to percentage
    print(f"Mean Relative Error (MRE): {mre:.2f}%")

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((Y_test - y_pred)**2))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Calculate Mean Absolute Scaled Error (MASE)
    mean_absolute_error = np.mean(np.abs(Y_test - y_pred))
    naive_errors = np.abs(Y_test[1:] - Y_test[:-1])
    mase = mean_absolute_error / np.mean(naive_errors)
    print(f"Mean Absolute Scaled Error (MASE): {mase:.2f}")
    print()
    print("Specificity:", specificity)
    print("Sensitivity:", sensitivity)
    print("True Positive Rate (TPR):", tpr)
    print("False Positive Rate (FPR):", fpr)
    print()
    # Calculate overall PLR and NLR
    print("+LR", PLR)
    print("-LR", NLR)
    print()
    print("Overall Positive Likelihood Ratio (PLR):", overall_PLR)
    print("Overall Negative Likelihood Ratio (NLR):", overall_NLR)
    print()
    cm()
    aoc(model_1pred_prob)

except ValueError as e:
    print("An error occurred:", e)

