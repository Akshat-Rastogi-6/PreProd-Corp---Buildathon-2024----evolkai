import streamlit as st
import pandas as pd
from model_utils import switch_case, cm, pre_processing, run_model

def set_custom_css(theme):
    # Set background image URL
    background_image_url = "https://i.pinimg.com/originals/22/7c/3c/227c3c56c0d21b4cf8b4b00f0e55c2a0.jpg"  # Pinterest image URL

    if theme == "Dark":
        st.markdown(f"""
            <style>
            .main {{
                background-image: url({background_image_url});
                background-size: cover;
                background-position: center;
                color: white;
            }}
            .sidebar {{
                background-color: #1E1E1E;
                color: white;
            }}
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <style>
            .main {{
                background-image: url({background_image_url});
                background-size: cover;
                background-position: center;
                color: black;
            }}
            .sidebar {{
                background-color: #F0F0F0;
                color: black;
            }}
            </style>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="New Theme App", layout="wide")
    
    st.sidebar.title("Theme Customization")
    theme_option = st.sidebar.selectbox("Select Theme:", ["Light", "Dark"])

    set_custom_css(theme_option)

    st.title("New Theme App 🎨")
    upload_file = st.file_uploader("Upload your CSV file...", type=['csv'])
    
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write('**DataFrame from Uploaded CSV File:**')
        st.dataframe(df.head())

        columns = df.columns
        df = pre_processing(df, columns)
        df = pd.DataFrame(df, columns=columns)

        model_name = st.sidebar.selectbox("Select Machine Learning Model :", [
            "Random Forest Classifier","SVM","Decision Tree Classifier","Logistic Regression", 
            "Adaboost Classifier","Extra Trees Classifier","Gradient Boosting Classifier",
            "K-Nearest Neighbors Classifier", "Gaussian Naive Bayes Classifier",
            "Bernoulli Naive Bayes Classifier", "Multinomial Naive Bayes Classifier",
            "Passive Aggressive Classifier", "Ridge Classifier", "Lasso Classifier",
            "ElasticNet Classifier", "Bagging Classifier", "Stochastic Gradient Descent Classifier",
            "Perceptron", "Isolation Forest", "Principal Component Analysis (PCA)",
            "Linear Discriminant Analysis (LDA)", "Quadratic Discriminant Analysis (QDA)",
            "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier", "MLP Classifier"
        ])

        model = switch_case(model_name)
        run_model(df, model, model_name)

if __name__ == "__main__":
    main()
