import streamlit as st
import pandas as pd
from PIL import Image
from model_utils import switch_case, cm, pre_processing, run_model

def main():
    st.set_page_config(page_title="Accurate 🎯", layout="wide")
    background_image_url = "https://i.pinimg.com/564x/11/83/7c/11837c0ee094b5e12f33fef3d41a1efa.jpg"

    st.markdown(f"""
            <style>
            .main {{
                background-image: url({background_image_url});
                background-size: cover;
                background-position: center;
                color: white;
            }}
            .st-emotion-cache-1avcm0n {{
                display: flex;
                height: 80px;
                align-items: center;
                justify-content: space-between;
            }}

            .st-emotion-cache-1avcm0n {{
                background-color: rgb(14 17 23 / 0%);
            }}

            .st-emotion-cache-1cypcdb {{
                background-color: rgb(14 17 23 / 0%);
            }}

            </style>
            """, unsafe_allow_html=True)
    
   

    st.title("Accurate 🎯")
    upload_file = st.file_uploader("Upload your CSV file...", type=['csv'])
    
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write('*DataFrame from Uploaded CSV File:*')
        st.dataframe(df.head())

        columns = df.columns
        df = pre_processing(df, columns)
        df = pd.DataFrame(df, columns=columns)

        model_name = st.sidebar.selectbox("Select Machine Learning Model:", [
            "Random Forest Classifier", "SVM", "Decision Tree Classifier", "Logistic Regression",
            "Adaboost Classifier", "Extra Trees Classifier", "Gradient Boosting Classifier",
            "K-Nearest Neighbors Classifier", "Gaussian Naive Bayes Classifier",
            "Bernoulli Naive Bayes Classifier", "Multinomial Naive Bayes Classifier",
            "Passive Aggressive Classifier", "Ridge Classifier", "Lasso Classifier",
            "ElasticNet Classifier", "Bagging Classifier", "Stochastic Gradient Descent Classifier",
            "Perceptron", "Isolation Forest", "Principal Component Analysis (PCA)",
            "Linear Discriminant Analysis (LDA)", "Quadratic Discriminant Analysis (QDA)",
            "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier", "MLP Classifier"
        ])

        hyperparameters = {}
        # Define hyperparameters based on the selected model
        if model_name in ["Random Forest Classifier", "SVM", "Decision Tree Classifier", "Logistic Regression",
                           "Adaboost Classifier", "Extra Trees Classifier", "Gradient Boosting Classifier",
                           "K-Nearest Neighbors Classifier", "Passive Aggressive Classifier", "Bagging Classifier",
                           "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier", "MLP Classifier"]:
            st.sidebar.subheader("Hyperparameters")
            if model_name == "Random Forest Classifier":
                hyperparameters['n_estimators'] = st.sidebar.number_input('Number of Estimators:', min_value=1, value=100)
                hyperparameters['max_depth'] = st.sidebar.number_input('Max Depth:', min_value=1, value=None)
            elif model_name == "SVM":
                hyperparameters['C'] = st.sidebar.number_input('C:', min_value=0.01, value=1.0)
                hyperparameters['kernel'] = st.sidebar.selectbox('Kernel:', ['linear', 'poly', 'rbf', 'sigmoid'])
            elif model_name == "Decision Tree Classifier":
                hyperparameters['max_depth'] = st.sidebar.number_input('Max Depth:', min_value=1, value=None)
                hyperparameters['min_samples_split'] = st.sidebar.number_input('Min Samples Split:', min_value=2, value=2)
            elif model_name == "Logistic Regression":
                hyperparameters['C'] = st.sidebar.number_input('C:', min_value=0.01, value=1.0)
                hyperparameters['solver'] = st.sidebar.selectbox('Solver:', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
            elif model_name == "Adaboost Classifier":
                hyperparameters['n_estimators'] = st.sidebar.number_input('Number of Estimators:', min_value=1, value=50)
            elif model_name == "Extra Trees Classifier":
                hyperparameters['n_estimators'] = st.sidebar.number_input('Number of Estimators:', min_value=1, value=100)
                hyperparameters['max_depth'] = st.sidebar.number_input('Max Depth:', min_value=1, value=None)
            elif model_name == "Gradient Boosting Classifier":
                hyperparameters['n_estimators'] = st.sidebar.number_input('Number of Estimators:', min_value=1, value=100)
                hyperparameters['learning_rate'] = st.sidebar.number_input('Learning Rate:', min_value=0.001, value=0.1)
            elif model_name == "K-Nearest Neighbors Classifier":
                hyperparameters['n_neighbors'] = st.sidebar.number_input('Number of Neighbors:', min_value=1, value=5)
            elif model_name == "Passive Aggressive Classifier":
                hyperparameters['C'] = st.sidebar.number_input('C:', min_value=0.01, value=1.0)
            elif model_name == "Bagging Classifier":
                hyperparameters['n_estimators'] = st.sidebar.number_input('Number of Estimators:', min_value=1, value=10)
            elif model_name == "XGBoost Classifier":
                hyperparameters['n_estimators'] = st.sidebar.number_input('Number of Estimators:', min_value=1, value=100)
                hyperparameters['learning_rate'] = st.sidebar.number_input('Learning Rate:', min_value=0.001, value=0.1)
            elif model_name == "LightGBM Classifier":
                hyperparameters['n_estimators'] = st.sidebar.number_input('Number of Estimators:', min_value=1, value=100)
                hyperparameters['learning_rate'] = st.sidebar.number_input('Learning Rate:', min_value=0.001, value=0.1)
            elif model_name == "CatBoost Classifier":
                hyperparameters['iterations'] = st.sidebar.number_input('Iterations:', min_value=1, value=100)
                hyperparameters['learning_rate'] = st.sidebar.number_input('Learning Rate:', min_value=0.001, value=0.1)
            elif model_name == "MLP Classifier":
                hyperparameters['hidden_layer_sizes'] = st.sidebar.text_input('Hidden Layer Sizes:', value='(100,)') # example: (100,)
                hyperparameters['activation'] = st.sidebar.selectbox('Activation:', ['identity', 'logistic', 'tanh', 'relu'])
                hyperparameters['solver'] = st.sidebar.selectbox('Solver:', ['lbfgs', 'sgd', 'adam'])

        model = switch_case(model_name, hyperparameters)
        if model:
            run_model(df, model, model_name)

if __name__ == "__main__":
    main()
