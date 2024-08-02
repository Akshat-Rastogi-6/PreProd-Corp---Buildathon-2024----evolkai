from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import streamlit as st
import pandas as pd
import numpy as np
from reportlab.lib.units import inch
from sklearn.calibration import LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, 
                              GradientBoostingClassifier, IsolationForest, 
                              RandomForestClassifier, StackingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (ElasticNet, Lasso, LogisticRegression, 
                                   PassiveAggressiveClassifier, Perceptron, 
                                   RidgeClassifier, SGDClassifier)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import (BernoulliNB, GaussianNB, MultinomialNB)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, 
                             f1_score, precision_score, recall_score, 
                             roc_auc_score, roc_curve, mean_squared_error)
from xgboost import XGBClassifier

# Function to choose between different models
def switch_case(argument):
    switcher = {
        "Random Forest Classifier": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Adaboost Classifier": AdaBoostClassifier(),
        "Extra Trees Classifier": ExtraTreeClassifier(),
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
        "MLP Classifier": MLPClassifier(),
        "Stacking Classifier": StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier()),
                ('svc', SVC(probability=True)),
                ('gb', GradientBoostingClassifier())
            ],
            final_estimator=LogisticRegression()
        )
    }
    return switcher.get(argument)

# Generate confusion matrix
def cm(y_test, y_pred):
    mdl_cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(13, 12))
    sns.heatmap(mdl_cm, annot=True)
    plt.savefig('uploads/confusion_matrix.jpg', format="jpg", dpi=300)
    st.image("uploads/confusion_matrix.jpg", caption="Confusion Matrix of your Data", width=600)
    return 'uploads/confusion_matrix.jpg'

# Generate PDF report
def generate_pdf(model_name, eval_mat, target_column, split_size, accuracy, confusion_matrix_image_path):
    pdf_path = "uploads/model_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.drawString(1 * inch, height - 1 * inch, "Model Evaluation Report")
    
    # Model Information
    c.drawString(1 * inch, height - 1.5 * inch, f"Model Selected: {model_name}")
    c.drawString(1 * inch, height - 2 * inch, f"Evaluation Metric: {eval_mat}")
    c.drawString(1 * inch, height - 2.5 * inch, f"Target Column: {target_column}")
    c.drawString(1 * inch, height - 3 * inch, f"Test Split Size: {split_size}")
    c.drawString(1 * inch, height - 3.5 * inch, f"Model Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    c.drawString(1 * inch, height - 4 * inch, "Confusion Matrix:")
    c.drawImage(confusion_matrix_image_path, 1 * inch, height - 8 * inch, width=6 * inch, height=4 * inch)

    c.save()
    return pdf_path

# Pre-process data
def pre_processing(df, columns):
    encoder = LabelEncoder()
    # Iterate through each column in the dataframe
    for column in df.columns:
        # Check if the column contains string values
        if df[column].dtype == 'object':
            # Fit label encoder and transform the column
            df[column] = encoder.fit_transform(df[column])

    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)

    return df
    
# Run model and evaluation
def run_model(df, model, model_name):
    st.subheader(model_name)
    
    target_column = st.text_input("Enter your target column : ")

    # Prepare data
    if target_column != "":
        X = df.drop(columns=[target_column])
        y = df[target_column]
        testing_size = st.text_input("Enter the test splitting size : ")
        if testing_size != "":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testing_size), random_state=42)

            # Train model
            model.fit(X_train, y_train)

            # Save the trained model
            model_filename = 'trained_model.pkl'
            joblib.dump(model, model_filename)
            st.write(f"Model saved as {model_filename}")

            # Provide a download link
            with open(model_filename, "rb") as file:
                st.download_button(label="Download Model", data=file, file_name=model_filename)

            # Make predictions
            testing_file = st.sidebar.file_uploader("Upload your testing CSV file...", type=['csv'])

            if testing_file is not None:
                test = pd.read_csv(testing_file)
                test_columns = test.columns

                test = pre_processing(test, test_columns)
                test = pd.DataFrame(test, columns=test_columns)

                y_pred = model.predict(test)

            y_pred = model.predict(X_test)
            
            eval_mat = st.sidebar.selectbox("Select your Evaluation Matrix :", ["Accuracy", "Precision", "Recall(Sensitivity)", "F1 Score", "Roc AUC Score", "RMSE"])

            result = ""
            if eval_mat == "Accuracy":
                result = str(accuracy_score(y_test, y_pred))
                st.write("Accuracy :", float(result))

            if eval_mat == "Precision":
                result = str(precision_score(y_test, y_pred, average='weighted'))
                st.write("Precision:", float(result))

            if eval_mat == "Recall(Sensitivity)":
                result = str(recall_score(y_test, y_pred, average='weighted'))
                st.write("Recall(Sensitivity):", float(result))

            if eval_mat == "F1 Score":
                result = str(f1_score(y_test, y_pred, average='weighted'))
                st.write("F1 Score :", float(result))
            
            if eval_mat == "Roc AUC Score":
                result = str(roc_auc_score(y_test, y_pred))
                st.write("Roc AUC Score :", float(result))
                
            if eval_mat == "RMSE":
                result = str(mean_squared_error(y_test, y_pred))
                st.write("RMSE:", float(result))

            if result != "":
                curves = st.sidebar.selectbox("Select the metrics you want to see : ", ["Confusion Matrix", "ROC Curve"])
                if curves == "Confusion Matrix":
                    confusion_matrix_image_path = cm(y_test, y_pred)
                if curves == "ROC Curve":
                    y_arr = np.array(y)
                    unique_classes, counts = np.unique(y_arr, return_counts=True)
                    n_classes = len(unique_classes)
                    y_train_bin = label_binarize(y_train, classes=range(n_classes))
                    classifier = OneVsRestClassifier(model)
                    classifier.fit(X_train, y_train_bin)

                    y_score = classifier.predict_proba(X_test)
                    aoc(y_score, n_classes, y_test)

            if eval_mat == "Accuracy" and curves == "Confusion Matrix":
                pdf_path = generate_pdf(model_name, eval_mat, target_column, testing_size, float(result), confusion_matrix_image_path)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(label="Download PDF Report", data=pdf_file, file_name="model_report.pdf")

def aoc(y_score, n_classes, y_test):   
    y_test_np = y_test.to_numpy() 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_np == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiclass Data')
    plt.legend(loc="lower right")
    plt.savefig('uploads/roc.jpg', format="jpg", dpi=300)
    st.image("uploads/roc.jpg", caption="ROC of your Data", width=600)

# Main function
def main():
    st.title("Accurate 🎯")
    query_params = st.query_params
    user_name = query_params.get("user", ["Guest"])

    st.write(f"Welcome, {user_name}!")

    upload_file = st.file_uploader("Upload your CSV file...", type=['csv'])
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write('**DataFrame from Uploaded CSV File:**')
        st.dataframe(df.head())
        
        columns = df.columns
        df = pre_processing(df, columns)

        df = pd.DataFrame(df, columns=columns)

        model_name = st.sidebar.selectbox("Select Machine Learning Model :", ["Random Forest Classifier","SVM","Decision Tree Classifier","Logistic Regression", "Adaboost Classifier","Extra Trees Classifier","Gradient Boosting Classifier","K-Nearest Neighbors Classifier", "Gaussian Naive Bayes Classifier", "Bernoulli Naive Bayes Classifier", "Multinomial Naive Bayes Classifier", "Passive Aggressive Classifier", "Ridge Classifier", "Lasso Classifier", "ElasticNet Classifier", "Bagging Classifier", "Stochastic Gradient Descent Classifier", "Perceptron", "Isolation Forest", "Principal Component Analysis (PCA)", "Linear Discriminant Analysis (LDA)", "Quadratic Discriminant Analysis (QDA)", "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier", "MLP Classifier", "Stacking Classifier"])

        model = switch_case(model_name)
        run_model(df, model, model_name)

if __name__ == "__main__":
    main()
