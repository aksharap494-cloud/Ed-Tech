import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="EdTech Lead Analysis", layout="wide")

st.title("📊 EdTech Lead Analysis & Modeling")
st.markdown("Streamlit app based on your Jupyter Notebook")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("leads_basic_details.csv")

df = load_data()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Dataset", "EDA", "Preprocessing", "Model Training", "Final Results"]
)

st.sidebar.header("Target Selection")
target_col = st.sidebar.selectbox(
    "Select Target Column (Required for ML)",
    df.columns
)

# --------------------------------------------------
# Dataset View
# --------------------------------------------------
if section == "Dataset":
    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("🎯 Selected Target Column")
    st.success(target_col)

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif section == "EDA":
    st.subheader("📈 Exploratory Data Analysis")

    col = st.selectbox("Select column for visualization", df.columns)
    fig, ax = plt.subplots()
    df[col].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
elif section == "Preprocessing":
    st.subheader("🛠️ Data Preprocessing")

    df_processed = df.copy()

    # Drop ID columns
    for col in df_processed.columns:
        if "id" in col.lower():
            df_processed.drop(col, axis=1, inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    for col in df_processed.select_dtypes(include="object").columns:
        df_processed[col] = le.fit_transform(df_processed[col])

    st.success("✅ Preprocessing Completed")
    st.write(df_processed.head())

# --------------------------------------------------
# Model Training
# --------------------------------------------------
elif section == "Model Training":
    st.subheader("🤖 Train Machine Learning Models")

    df_model = df.copy()

    # Drop ID columns
    for col in df_model.columns:
        if "id" in col.lower():
            df_model.drop(col, axis=1, inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = le.fit_transform(df_model[col])

    if target_col not in df_model.columns:
        st.error("Target column not found after preprocessing")
        st.stop()

    X = df_model.drop(target_col, axis=1)
    y = df_model[target_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_name = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest", "SVM", "KNN"]
    )

    if st.button("🚀 Train Model"):
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "SVM":
            model = SVC()
        else:
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# --------------------------------------------------
# Final Results
# --------------------------------------------------
elif section == "Final Results":
    st.subheader("🏆 Model Comparison")

    df_model = df.copy()

    for col in df_model.columns:
        if "id" in col.lower():
            df_model.drop(col, axis=1, inplace=True)

    le = LabelEncoder()
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop(target_col, axis=1)
    y = df_model[target_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)

    results_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
    st.table(results_df.sort_values(by="Accuracy", ascending=False))
